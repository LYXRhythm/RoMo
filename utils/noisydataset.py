# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import random
from logging import getLogger

import cv2
from PIL import ImageFilter, Image
import numpy as np
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.utils.data as data
import scipy.io as sio
import torch
from numpy.testing import assert_array_almost_equal
from utils.cluster import cluster_acc

logger = getLogger()

def multiclass_noisify(y, P, random_state=0):
    """ Flip classes according to transition probability matrix T.
    It expects a number between 0 and the number of classes - 1.
    """

    assert P.shape[0] == P.shape[1]
    assert np.max(y) < P.shape[0]

    # row stochastic matrix
    assert_array_almost_equal(P.sum(axis=1), np.ones(P.shape[1]))
    assert (P >= 0.0).all()

    m = y.shape[0]
    new_y = y.copy()
    flipper = np.random.RandomState(random_state)

    for idx in np.arange(m):
        i = y[idx]
        flipped = flipper.multinomial(1, P[i, :], 1)[0]
        new_y[idx] = np.where(flipped == 1)[0]

    return new_y

def other_class(n_classes, current_class):
    """
    Returns a list of class indices excluding the class indexed by class_ind
    :param nb_classes: number of classes in the task
    :param class_ind: the class index to be omitted
    :return: one random class that != class_ind
    """
    if current_class < 0 or current_class >= n_classes:
        error_str = "class_ind must be within the range (0, nb_classes - 1)"
        raise ValueError(error_str)

    other_class_list = list(range(n_classes))
    other_class_list.remove(current_class)
    other_class_ = np.random.choice(other_class_list)
    return other_class_

class cross_modal_dataset(data.Dataset):
    def __init__(self, dataset_name, noisy_mode, noisy_ratio, mode, modal_list, image_file_list, image_transform=None,
                class_num=10, root_dir='data/', noise_file=None, pred=False, probability=[], log=''):
        self.dataset_name = dataset_name
        self.r = noisy_ratio # noise ratio
        self.noisy_mode = noisy_mode
        self.mode = mode
        self.modal_list = modal_list
        self.class_num = class_num
        self.loader = self.pil_loader
        self.image_transform = image_transform
        self.train_data_path = []
        self.train_data = []
        self.train_label = []
        self.noisy_label = []

        for ii in range(len(image_file_list)):
            datas, labels = self.make_dataset_nolist(image_file_list[ii])
            self.train_data_path.append(datas)
            self.train_label.append(labels)

        temp = []
        for ii in range(len(self.modal_list)):
            if self.modal_list[ii] == "RGBImg":
                for jj in range(len(self.train_data_path[ii])):
                    if self.image_transform==None:
                        temp.append(np.array(self.loader(self.train_data_path[ii][jj])))
                    else:
                        temp.append(self.image_transform(self.loader(self.train_data_path[ii][jj])))
                temp = torch.tensor([item.cpu().detach().numpy() for item in temp])
                print(self.dataset_name, ": ", self.mode, " ", self.modal_list[ii], " ", temp.shape)
                self.train_data.append(temp)
                temp = []
            elif self.modal_list[ii] == "GrayImg":
                for jj in range(len(self.train_data_path[ii])):
                    if self.image_transform==None:
                        temp.append(np.array(self.loader(self.train_data_path[ii][jj])))
                    else:
                        temp.append(self.image_transform(self.loader(self.train_data_path[ii][jj])))
                temp = torch.tensor([item.cpu().detach().numpy() for item in temp])
                print(self.dataset_name, ": ", self.mode, " ", self.modal_list[ii], " ", temp.shape)
                self.train_data.append(temp)
                temp = []
            elif self.modal_list[ii] == "Mesh":
                for jj in range(len(self.train_data_path[ii])):
                    tt = np.load(self.train_data_path[ii][jj])
                    self.train_data.append(tt)

            elif self.modal_list[ii] == "PointCloud":
                for jj in range(len(self.train_data_path[ii])):
                    tt = self.translate_pointcloud(self.random_down_sample(np.load(self.train_data_path[ii][jj]), 1024))
                    temp.append(tt)
                temp = torch.tensor(np.array(temp))
                temp = temp.permute(0, 2, 1)
                print(self.dataset_name, ": ", self.mode, " ", self.modal_list[ii], " ", temp.shape)
                self.train_data.append(temp)
                temp = []
            elif self.modal_list[ii] == "Txt":
                pass
            else:
                raise ValueError("Error Modal")

        if noisy_mode=='None' or noisy_mode==None:
            self.noisy_label = np.array(self.train_label)
        elif noisy_mode=='sym' or noisy_mode=='asym':
            self.noisy_label = self.transform_noise_label(self.train_label)
        else:
            raise ValueError("Error Noisy Mode")
        
        self.noisy_label = torch.tensor(self.noisy_label)

    def __getitem__(self, index):
        return [self.train_data[v][index] for v in range(len(self.train_data))], \
                [self.noisy_label[v][index] for v in range(len(self.train_data))], \
                 [self.train_data_path[v][index] for v in range(len(self.train_data))], index
                  
    def __len__(self):
        return len(self.train_data[0])
    
    def pil_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')
    
    def make_dataset_nolist(self, image_list):
        with open(image_list) as f:
            image_index = [x.split(' ')[0] for x in f.readlines()]
        with open(image_list) as f:
            label_list = []
            selected_list = []
            for ind, x in enumerate(f.readlines()):
                label = x.split(' ')[1].strip()
                label_list.append(int(label))
                selected_list.append(ind)
            image_index = np.array(image_index)
            label_list = np.array(label_list)
        image_index = image_index[selected_list]
        # image_index = image_index[0:200]
        # label_list = label_list[0:200]
        return image_index, label_list
    
    def transform_noise_label(self, gt_label):
        noise_label = []
        inx = np.arange(self.class_num)
        np.random.shuffle(inx)
        transition = {i: i for i in range(self.class_num)}
        half_num = int(self.class_num // 2)
        for i in range(half_num):
            transition[inx[i]] = int(inx[half_num + i])
        for v in range(len(gt_label)):
            noise_label_tmp = []
            data_num = gt_label[v].shape[0]
            idx = list(range(data_num))
            random.shuffle(idx)
            num_noise = int(self.r * data_num)
            noise_idx = idx[:num_noise]
            for i in range(data_num):
                if i in noise_idx:
                    if self.noisy_mode == 'sym':
                        noiselabel = int(random.randint(0, 9))
                        noise_label_tmp.append(noiselabel)
                    elif self.noisy_mode == 'asym':
                        noiselabel = transition[gt_label[v][i]]
                        noise_label_tmp.append(noiselabel)
                else:
                    noise_label_tmp.append(int(gt_label[v][i]))
            noise_label.append(noise_label_tmp)
        noise_label = np.array(noise_label)
        for i in range(noise_label.shape[0]):
            for j in range(noise_label.shape[1]):
                noise_label[i][j] = noise_label[i][j] % self.class_num
        return noise_label

    def random_down_sample(self, cloud, sample_points):
        row_rand_array = np.arange(cloud.shape[0])
        np.random.shuffle(row_rand_array)
        row_rand = cloud[row_rand_array[0:sample_points], :]
        return row_rand

    def translate_pointcloud(self, pointcloud):
        xyz1 = np.random.uniform(low=2./3., high=3./2., size=[3])
        xyz2 = np.random.uniform(low=-0.2, high=0.2, size=[3])
        
        translated_pointcloud = np.add(np.multiply(pointcloud, xyz1), xyz2).astype('float32')
        return translated_pointcloud

class RandomGaussianBlur(object):
    def __call__(self, img):
        do_it = np.random.rand() > 0.5
        if not do_it:
            return img
        sigma = np.random.rand() * 1.9 + 0.1
        return cv2.GaussianBlur(np.asarray(img), (23, 23), sigma)

class PILRandomGaussianBlur(object):
    def __init__(self, p=0.5, radius_min=0.1, radius_max=2.):
        self.prob = p
        self.radius_min = radius_min
        self.radius_max = radius_max

    def __call__(self, img):
        do_it = np.random.rand() <= self.prob
        if not do_it:
            return img

        return img.filter(
            ImageFilter.GaussianBlur(
                radius=random.uniform(self.radius_min, self.radius_max)
            )
        )

def get_color_distortion(s=1.0):
    color_jitter = transforms.ColorJitter(0.8*s, 0.8*s, 0.8*s, 0.2*s)
    rnd_color_jitter = transforms.RandomApply([color_jitter], p=0.8)
    rnd_gray = transforms.RandomGrayscale(p=0.2)
    color_distort = transforms.Compose([rnd_color_jitter, rnd_gray])
    return color_distort
