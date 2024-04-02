import torch
import torch.nn as nn
import torch.nn.functional as F

class CELoss(torch.nn.Module):
    def __init__(self, ):
        super(CELoss, self).__init__()
        self.cross_entropy = torch.nn.CrossEntropyLoss()

    def forward(self, preds, labels):
        ce = self.cross_entropy(preds, labels.long())
        return ce

class RCLLLoss(nn.Module):
    def __init__(self, alpha=0.65, num_classes=10, eps=1e-7, scale=1.):
        super(RCLLLoss, self).__init__()
        self.alpha = alpha
        self.num_classes = num_classes
        self.eps = eps
        self.scale = scale
        
    def forward(self, pred, labels):
        K = len(set(labels))
        threshold = 1.0 / K
        pred = F.softmax(pred, dim=1)
        label_one_hot = F.one_hot(labels, self.num_classes).float().to(pred.device)
        indices_clip = torch.where(pred <= threshold)[0]
        loss_clip = torch.pow((- torch.log(torch.sum(label_one_hot[indices_clip] * threshold, dim=1))), self.alpha) * (1 - torch.sum(label_one_hot[indices_clip] * threshold, dim=1))
        
        indices_normal = torch.where(pred > threshold)[0]
        loss_normal = torch.pow((- torch.log(torch.sum(label_one_hot[indices_normal] * pred[indices_normal], dim=1))), self.alpha) * (1 - torch.sum(label_one_hot[indices_normal] * pred[indices_normal], dim=1))
        
        return (loss_clip.mean()+loss_normal.mean())* self.scale
    
class CrossModalLoss(nn.Module):
    def __init__(self, tau=0.22, num_classes=40, modal_num=2, feat_dim=512, warmup=True, centers=None):
        super(CrossModalLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.warmup = warmup
        self.tau = tau
        self.modal_num = modal_num

    def forward(self, x):
        batch_size = x.size(0)
        x = F.normalize(x, p=2, dim=1)
        
        batch_size = x.shape[0] // self.modal_num
        sim = x.mm(x.t())
        sim = (sim / self.tau).exp()
        sim = sim - sim.diag().diag()

        sim_sum1 = sum([sim[:, v * batch_size: (v + 1) * batch_size] for v in range(self.modal_num)])

        diag1 = torch.cat([sim_sum1[v * batch_size: (v + 1) * batch_size].diag() for v in range(self.modal_num)])
        loss1 = -(diag1 / sim.sum(1)).log().mean()

        sim_sum2 = sum([sim[v * batch_size: (v + 1) * batch_size] for v in range(self.modal_num)])
        diag2 = torch.cat([sim_sum2[:, v * batch_size: (v + 1) * batch_size].diag() for v in range(self.modal_num)])
        loss2 = -(diag2 / sim.sum(1)).log().mean()
        return loss1 + loss2