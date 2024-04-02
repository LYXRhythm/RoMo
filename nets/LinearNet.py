import torch
from torch import nn
import torch.nn.functional as F

class LinearAverage(nn.Module):
    def __init__(self, inputSize, outputSize, T=0.05, momentum=0.5):
        super(LinearAverage, self).__init__()
        self.nLem = outputSize
        # self.unigrams = torch.ones(self.nLem)
        # self.multinomial = AliasMethod(self.unigrams)
        # self.multinomial.cuda()
        self.momentum = momentum
        self.register_buffer('params', torch.tensor([T, momentum]))
        self.register_buffer('memory', torch.zeros(outputSize, inputSize))
        self.register_buffer('targets_memory', torch.zeros(outputSize, ))
        self.T = T
        self.memory = self.memory.cuda()
        self.memory_first = True

    def forward(self, x, use_softmax=True):
        out = torch.mm(x, self.memory.t())
        if use_softmax:
            out = out/self.T
        else:
            out = torch.exp(torch.div(out, self.T))
            Z_l = (out.mean() * self.nLem).clone().detach().item()
            out = torch.div(out, Z_l).contiguous()
        return out

    def update_weight(self, features, index):
        weight_pos = self.memory.index_select(0, index.data.view(-1)).resize_as_(features)
        weight_pos.mul_(self.momentum)
        weight_pos.add_(torch.mul(features.data, 1 - self.momentum))

        w_norm = weight_pos.pow(2).sum(1, keepdim=True).pow(0.5)
        updated_weight = weight_pos.div(w_norm)
        self.memory.index_copy_(0, index, updated_weight)
        self.memory = F.normalize(self.memory)
        
        memory_size_bytes = self.memory.element_size() * self.memory.nelement()
        memory_size_mb = memory_size_bytes / (1024 ** 2)
        features_size_bytes = features.element_size() * features.nelement()
        features_size_mb = features_size_bytes / (1024 ** 2)

    def set_weight(self, features, index):
        self.memory.index_select(0, index.data.view(-1)).resize_as_(features)

class HeadNet(nn.Module):
    def __init__(self, output_dim, num_classes):
        super(HeadNet, self).__init__()
        self.output_dim = output_dim
        self.num_classes = num_classes

        self.head = nn.Sequential(*[nn.Linear(self.output_dim, 128), 
                                    nn.ReLU(), 
                                    nn.Linear(128, self.num_classes)])

    def forward(self, multi_feature):
        multi_pred = []
        for i in range(len(multi_feature)):
            multi_pred.append(self.head(multi_feature[i]))
        # img_pred = self.head(img_feat)
        # pt_pred = self.head(pt_feat)

        return multi_pred