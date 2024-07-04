# RoMo
RoMo: Robust Unsupervised Multimodal Learning with Noisy Pseudo Labels (IEEE Transaction on Image Processing, PyTorch Code)

Authors: Yongxiang Li, Yang Qin, Yuan Sun, Dezhong Peng, Xi Peng and Peng Hu

## Abstract
The rise of the metaverse and the increasing volume of heterogeneous 2D and 3D data have led to a growing demand for cross-modal retrieval, which allows users to query semantically relevant data across different modalities. Existing methods heavily rely on class labels to bridge semantic correlations, but it is expensive or even impossible to collect large-scale well-labeled data in practice, thus making unsupervised learning more attractive and practical. However, unsupervised cross-modal learning is challenging to bridge semantic correlations across different modalities due to the lack of label information, which inevitably leads to unreliable discrimination. Based on the observations, we reveal and study a novel problem in this paper, namely unsupervised cross-modal learning with noisy pseudo labels. To address this problem, we propose a 2D-3D unsupervised multimodal learning framework that harnesses multimodal data. Our framework consists of three key components: 1) Self-matching Supervision Mechanism (SSM) warms up the model to encapsulate discrimination into the representations in a self-supervised learning manner. 2) Robust Discriminative Learning (RDL) further mines the discrimination from the learned imperfect predictions after warming up. To tackle the noise in the predicted pseudo labels, RDL leverages a novel Robust Concentrating Learning Loss (RCLL) to alleviate the influence of the uncertain samples, thus embracing robustness against noisy pseudo labels. 3) Modality-invariance Learning Mechanism (MLM) minimizes the cross-modal discrepancy to enforce SSM and RDL to produce common representations. We perform comprehensive experiments on four 2D-3D multimodal datasets, comparing our method against 14 state-of-the-art approaches, thereby demonstrating its effectiveness and superiority.

## Framework
![pipline](./figs/pipline_figure.png)

## Requirements
```bash
pip install requirements.txt
```

## Dataset
[Kaggle-3D MNIST](https://www.kaggle.com/datasets/daavoo/3d-mnist) data is currently available for your training and testing. 

Here, we provide a processed version of 3D MNIST (https://drive.google.com/file/d/1_qk06tW7HAPmnHCdfgoIX__RgJMxRRyY/view?usp=drive_link) to match with our code. Download, unzip and put it in the ./dataset/

## Pre-Trained Model for Feature Extraction Network
### 2D Modality Pre-trained Model:
2D modaility (RGB and GARY) Pre-Trained Model will automatic download.

### 3D Modality Pre-trained Model:
1. DGCNN (for POINT CLOUD) in the 3D modaility Pre-trained Model can access in the https://drive.google.com/file/d/1lnvyf2Gh5Dy19yzQ6cx-a9IaX_BChpae/view?usp=drive_link.
Download and put it in the ./pretrained/

2. MeshNet (for MESH) in the 3D modaility Pre-trained Model can access in the https://github.com/iMoonLab/MeshNet.

## Quickly Training
### PLA Stage:
```python
python train_step1.py
```
After train_step1, you will get the pseudo-labels (***mnist3d_PointCloud_train_list_pseudo_labelling.txt***, ***mnist3d_RGBImgs_train_list_pseudo_labelling.txt***) in the current directory. 

For your convenience, we have provided an example pseudo label file (in root directory ***mnist3d_PointCloud_train_list_pseudo_labelling.txt***, ***mnist3d_RGBImgs_train_list_pseudo_labelling.txt***) for you to run ***train_step2.py*** directly.

### RDL Stage:
```python
python train_step2.py
```

## Thanks
Thanks for the reference and assistance provided by the following work for this code repository.
RONO: https://github.com/penghu-cs/RONO
MRL:  https://github.com/penghu-cs/MRL

## Citation
If you find this work useful in your research, please consider citing:
@article{li2024romo,
  title={RoMo: Robust Unsupervised Multimodal Learning with Noisy Pseudo Labels},
  author={Li, Yongxiang and Qin Yang and Sun, Yuan and Peng, Dezhong and Peng, Xi and Hu, Peng},
  journal={IEEE Transactions on Image Processing},
  year={2024},
  publisher={IEEE}
}

