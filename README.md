# RoMo
Yongxiang Li, Yang Qin, Yuan Sun, Dezhong Peng, Xi Peng and Peng Hu, RoMo: Robust Unsupervised Multimodal Learning with Noisy Pseudo Labels (PyTorch Code)

## Abstract
The rise of the metaverse and the increasing volume of heterogeneous 2D and 3D data have led to a growing demand for cross-modal retrieval, which allows users to query semantically relevant data across different modalities. Existing methods heavily rely on class labels to bridge semantic correlations, but it is expensive or even impossible to collect large-scale well-labeled data in practice, thus making unsupervised learning more attractive and practical. However, unsupervised cross-modal learning is challenging to bridge semantic correlations across different modalities due to the lack of label information, which inevitably leads to unreliable discrimination. Based on the observations, we reveal and study a novel problem in this paper, namely unsupervised cross-modal learning with noisy pseudo labels. To address this problem, we propose a 2D-3D unsupervised multimodal learning framework that harnesses multimodal data. Our framework consists of three key components: 1) Self-matching Supervision Mechanism (SSM) warms up the model to encapsulate discrimination into the representations in a self-supervised learning manner. 2) Robust Discriminative Learning (RDL) further mines the discrimination from the learned imperfect predictions after warming up. To tackle the noise in the predicted pseudo labels, RDL leverages a novel Robust Concentrating Learning Loss (RCLL) to alleviate the influence of the uncertain samples, thus embracing robustness against noisy pseudo labels. 3) Modality-invariance Learning Mechanism (MLM) minimizes the cross-modal discrepancy to enforce SSM and RDL to produce common representations. We perform comprehensive experiments on four 2D-3D multimodal datasets, comparing our method against 14 state-of-the-art approaches, thereby demonstrating its effectiveness and superiority.

## Framework
![pipline](./figs/pipline_figure.png)

## Requirements
```bash
pip install requirements.txt
```

## Dataset
3D MNIST dataset data is currently available for your training and testing. 

If you use raw data [Kaggle-3D MNIST](https://www.kaggle.com/datasets/daavoo/3d-mnist) , suitable data augmentation can bring the performance of the method to a higher level. 

Here, we provide a processed version (https://drive.google.com/file/d/1_qk06tW7HAPmnHCdfgoIX__RgJMxRRyY/view?usp=drive_link) to match with our code. Download, unzip and put it in the ./dataset/

## Pre-Trained Model for Feature Extraction Network
2D modaility Pre-Trained Model will automatic download.

Dgcnn in the 3D modaility Pre-trained Model can access in the https://drive.google.com/file/d/1lnvyf2Gh5Dy19yzQ6cx-a9IaX_BChpae/view?usp=drive_link. Download and put it in the ./pretrained/

## Quickly Training
### PLA Stage:
```python
python train_step1.py
```
After train_step1, you will get the pseudo-labels (***mnist3d_PointCloud_train_list_pseudo_labelling.txt***, ***mnist3d_RGBImgs_train_list_pseudo_labelling.txt***) in the current directory. FOR YOUR CONVENIENCE, WE PROVIDED AN EXAMPLE PSEUDO LABEL FILE FOR YOUR DIRECTLY RUNNING STEP2.

### RDL Stage:
```python
python train_step2.py
```

## Thanks
Thanks for the reference and assistance provided by the following work for this code repository.
```
https://github.com/penghu-cs/MRL
@inproceedings{hu2021MRL,
   title={Learning Cross-Modal Retrieval with Noisy Labels},
   author={Peng Hu, Xi Peng, Hongyuan Zhu, Liangli Zhen, Jie Lin},
   booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
   pages={5403--5413},
   year={2021}
}
```
```
https://github.com/penghu-cs/RONO
@InProceedings{Feng_2023_CVPR,
    author    = {Feng, Yanglin and Zhu, Hongyuan and Peng, Dezhong and Peng, Xi and Hu, Peng},
    title     = {RONO: Robust Discriminative Learning With Noisy Labels for 2D-3D Cross-Modal Retrieval},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2023},
    pages     = {11610-11619}
}
```
