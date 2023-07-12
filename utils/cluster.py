# coding:utf-8
import numpy as np
from tqdm import tqdm
from faiss import Kmeans as faiss_Kmeans
from sklearn.cluster import KMeans
from sklearn.utils.linear_assignment_ import linear_assignment

import torch
from torch import nn
import torch.nn.functional as F

import warnings
warnings.filterwarnings("ignore")
from sklearn.cluster import KMeans

DEFAULT_KMEANS_SEED = 313

def TKmeans(features, k_classes=3, init_centroids=None):
    k_means = KMeans(n_clusters=k_classes, random_state=DEFAULT_KMEANS_SEED)
    k_means.fit(features)
    y_predict = k_means.predict(features)
    cluster_centers = k_means.cluster_centers_
    return y_predict, cluster_centers

class Kmeans(object):
    def __init__(self, k_list, data, epoch=0, init_centroids=None, frozen_centroids=False):
        """
        Performs many k-means clustering.
        Args:
            data (np.array N * dim): data to cluster
        """
        super().__init__()
        self.k_list = k_list
        self.data = data
        self.d = data.shape[-1]
        self.init_centroids = init_centroids
        self.frozen_centroids = frozen_centroids

        self.debug = False
        self.epoch = epoch + 1

    def compute_clusters(self):
        """compute cluster
        Returns:
            torch.tensor, list: clus_labels, centroids
        """
        data = self.data
        labels = []
        centroids = []

        tqdm_batch = tqdm(total=len(self.k_list), desc="[K-means]")
        for k_idx, each_k in enumerate(self.k_list):
            seed = k_idx * self.epoch + DEFAULT_KMEANS_SEED
            kmeans = faiss_Kmeans(self.d, each_k, niter=40, verbose=False,
                spherical=True, min_points_per_centroid=1, max_points_per_centroid=10000,
                gpu=True, seed=seed, frozen_centroids=self.frozen_centroids,
            )

            kmeans.train(data, init_centroids=self.init_centroids)

            _, I = kmeans.index.search(data, 1)
            labels.append(I.squeeze(1))
            C = kmeans.centroids
            centroids.append(C)

            tqdm_batch.update()
        tqdm_batch.close()
        labels = np.stack(labels, axis=0)

        return labels, centroids

def torch_kmeans(k_list, data, init_centroids=None, seed=0, frozen=False):
    if init_centroids is not None:
        init_centroids = init_centroids.cpu().numpy()
    km = Kmeans(
        k_list,
        data.cpu().detach().numpy(),
        epoch=seed,
        frozen_centroids=frozen,
        init_centroids=init_centroids,
    )
    clus_labels, centroids_npy = km.compute_clusters()
    clus_labels = torch.from_numpy(clus_labels).long().cuda()
    centroids = []
    for c in centroids_npy:
        centroids.append(torch.from_numpy(c).cuda())
    return clus_labels, centroids

def cluster_acc(y_true, y_pred):
    """
    Calculate clustering accuracy. Require scikit-learn installed
    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`
    # Return
        accuracy, in [0,1]
    """
    y_true = y_true.astype(np.int64)
    y_pred = y_pred.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    ind = linear_assignment(w.max() - w)
    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size