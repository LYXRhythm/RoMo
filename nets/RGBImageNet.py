import os
import torch
from torch import nn
from torch.nn import functional as F

import nets.resnet as resnet
from nets.LinearNet import LinearAverage

class UnsupervisedRGBImageNet(nn.Module):
    def __init__(self, args, memorySize):
        super(UnsupervisedRGBImageNet, self).__init__()
        self.args = args
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
        if args.backbone_2d == 'resnet18':
            self.backbone = resnet.resnet18(pretrained=args.pretretrained, low_dim=args.output_dim)
        elif args.backbone_2d == 'resnet50':
            self.backbone = resnet.resnet50(pretrained=args.pretretrained, low_dim=args.output_dim)
        elif args.backbone_2d == 'resnet101':
            self.backbone = resnet.resnet101(pretrained=args.pretretrained, low_dim=args.output_dim)
        else:
            raise NotImplementedError

        self.classifier = LinearAverage(inputSize=args.output_dim, outputSize=memorySize, T=args.membank_t, momentum=args.membank_m)
        self.self_top_layer = nn.Linear(args.output_dim, args.class_num, bias=False)

    def forward(self, x):
        emd = self.backbone(x)
        out = self.self_top_layer(emd)
        return out, emd

    def feature_ext(self, train_loader, v_th_views):
        self.backbone.eval()
        trainFeatures = self.classifier.memory.t()
        trainLabels = self.classifier.targets_memory.t()
    
        with torch.no_grad():
            for batch_idx, (batches, targets, batches_path, index) in enumerate(train_loader):
                batches, targets = batches[v_th_views].cuda(), targets[v_th_views].cuda()
                features = self.backbone(batches)
                features = F.normalize(features, dim=0)
                trainFeatures[:, batch_idx * self.args.train_batch_size:batch_idx * self.args.train_batch_size + self.args.train_batch_size] = features.data.t().cuda()
                trainLabels[batch_idx * self.args.train_batch_size:batch_idx * self.args.train_batch_size + self.args.train_batch_size] = targets.cuda()
        
        self.classifier.memory = trainFeatures.t().cuda()
        self.classifier.targets_memory = trainLabels.t().cuda()
        self.classifier.memory_first = False

class RGBImageNet(nn.Module):
    def __init__(self, args):
        super(RGBImageNet, self).__init__()
        self.args = args
        if args.backbone_2d == 'resnet18':
            self.backbone = resnet.resnet18(pretrained=args.pretretrained, low_dim=args.output_dim)
        elif args.backbone_2d == 'resnet50':
            self.backbone = resnet.resnet50(pretrained=args.pretretrained, low_dim=args.output_dim)
        elif args.backbone_2d == 'resnet101':
            self.backbone = resnet.resnet101(pretrained=args.pretretrained, low_dim=args.output_dim)
        else:
            raise NotImplementedError

    def forward(self, x):
        pred = self.backbone(x)
        return pred
