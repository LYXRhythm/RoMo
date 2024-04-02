# -*- coding=utf-8 -*-
import os
import numpy as np
import scipy.io as sio
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter

import nets as models
from paraser import step2_setting
import utils.utils as utils
import utils.losses as lossfunction
from utils.preprocess import *
from utils.bar_show import progress_bar
from utils.noisydataset import cross_modal_dataset

Image_transforms = {
    'train': transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
    ]),
}

def load_dict(model, path):
    chp = torch.load(path)
    state_dict = model.state_dict()
    for key in state_dict:
        if key in chp['model_state_dict']:
            state_dict[key] = chp['model_state_dict'][key]
    model.load_state_dict(state_dict)

def set_train():
    for v in range(n_view):
        multi_models[v].train()

def set_eval():
    for v in range(n_view):
        multi_models[v].eval()

def multiview_test(fea, lab):
    MAPs = np.zeros([n_view, n_view])
    val_dict = {}
    print_str = ''
    for i in range(n_view):
        for j in range(n_view):
            if i == j:
                continue
            MAPs[i, j] = utils.fx_calc_map_label(fea[j], lab[j], fea[i], lab[i], k=0, metric='cosine')[0]
            key = '%s2%s' % (args.views[i], args.views[j])
            val_dict[key] = MAPs[i, j]
            print_str = print_str + key + ': %.3f\t' % val_dict[key]
    return val_dict, print_str

def train(epoch):
    print('\nEpoch: %d / %d' % (epoch, args.max_epochs), file=print_log)
    print('\nEpoch: %d / %d' % (epoch, args.max_epochs))
    set_train()
    train_loss, loss_list, correct_list, total_list = 0., [0.] * n_view, [0.] * n_view, [0.] * n_view

    for batch_idx, (batches, targets, batches_path, index) in enumerate(train_loader):
        batches, targets = [batches[v].cuda() for v in range(n_view)], [targets[v].cuda() for v in range(n_view)]

        for v in range(n_view):
            multi_models[v].zero_grad()
        optimizer.zero_grad()

        outputs = [multi_models[v](batches[v]) for v in range(n_view)]
        preds = head_net(outputs)

        rb_losses = [rb_criterion(preds[v], targets[v]) for v in range(n_view)]
        rb_loss = sum(rb_losses)
        cross_loss = cross_criterion(torch.cat((outputs[0], outputs[1]), dim=0))

        # loss = args.lambda_rb*rb_loss + (1-args.lambda_crossmodal)*cross_loss
        loss = 0 + (1-args.lambda_crossmodal)*cross_loss

        if epoch >= 0:
            loss.backward()
            optimizer.step()
        train_loss += loss.item()

        for v in range(n_view):
            loss_list[v] += 0
            _, predicted = preds[v].max(1)
            total_list[v] += targets[v].size(0)
            acc = predicted.eq(targets[v]).sum().item()
            correct_list[v] += acc
        progress_bar(batch_idx, len(train_loader), 'Loss: %.3f | LR: %g'
                        % (train_loss / (batch_idx + 1), optimizer.param_groups[0]['lr']))
        print('Loss: %.3f | LR: %g'
                % (train_loss / (batch_idx + 1), optimizer.param_groups[0]['lr']), file=print_log)

def eval(data_loader, epoch, mode='test'):
    fea, lab, pred_score = [[] for _ in range(n_view)], [[] for _ in range(n_view)], [[] for _ in range(n_view)]
    correct_list, total_list = [0.] * n_view, [0.] * n_view
    with torch.no_grad():
        if sum([data_loader.dataset.train_data[v].shape[0] != data_loader.dataset.train_data[0].shape[0] for v in range(len(data_loader.dataset.train_data))]) == 0:
            for batch_idx, (batches, targets, batches_path, index) in enumerate(data_loader):
                batches, targets = [batches[v].cuda() for v in range(n_view)], [targets[v].cuda() for v in range(n_view)]
                outputs = [multi_models[v](batches[v]) for v in range(n_view)]
                pred, losses = [], []
                for v in range(n_view):
                    fea[v].append(outputs[v])
                    lab[v].append(targets[v])
                    pred = head_net(outputs)
                    pred_score[v].append(pred[v])
                    
                    _, predicted = pred[v].max(1)
                    total_list[v] += targets[v].size(0)
                    acc = predicted.eq(targets[v]).sum().item()
                    correct_list[v] += acc
        else:
            pred, losses = [], []
            for v in range(n_view):
                count = int(np.ceil(data_loader.dataset.train_data[v].shape[0]) / data_loader.batch_size)
                for ct in range(count):
                    batch, targets = torch.Tensor(data_loader.dataset.train_data[v][ct * data_loader.batch_size: (ct + 1) * data_loader.batch_size]).cuda(), torch.Tensor(data_loader.dataset.noise_label[v][ct * data_loader.batch_size: (ct + 1) * data_loader.batch_size]).long().cuda()
                    outputs = multi_models[v](batch)
                    fea[v].append(outputs)
                    lab[v].append(targets)
                    pred = head_net(outputs)
                    pred_score[v].append(pred[v])

                    _, predicted = pred[v].max(1)
                    total_list[v] += targets.size(0)
                    acc = predicted.eq(targets).sum().item()
                    correct_list[v] += acc

        fea = [torch.cat(fea[v]).cpu().detach().numpy() for v in range(n_view)]
        lab = [torch.cat(lab[v]).cpu().detach().numpy() for v in range(n_view)]
        pred_score = [torch.cat(pred_score[v]).cpu().detach().numpy() for v in range(n_view)]            

    return fea, lab, pred_score

def test(epoch):
    global best_acc
    set_eval()

    ## Calculate MAPs in val dataset 
    fea, gt_label, pred_score = eval(valid_loader, epoch, 'valid')
    MAPs = np.zeros([n_view, n_view])
    val_dict = {}
    print_val_str = 'Validation(mAP): '

    for i in range(n_view):
        for j in range(n_view):
            if i == j:
                continue
            MAPs[i, j] = utils.fx_calc_map_label(fea[j], gt_label[j], fea[i], gt_label[i], k=0, metric='cosine')[0]
            key = '%s2%s' % (args.views[i], args.views[j])
            val_dict[key] = MAPs[i, j]
            print_val_str = print_val_str + key +': %g\t' % val_dict[key]

    val_avg = MAPs.sum() / n_view / (n_view - 1.)
    val_dict['avg'] = val_avg
    print_val_str = print_val_str + 'Avg: %g' % val_avg
    summary_writer.add_scalars('Retrieval/valid', val_dict, epoch)
    print(print_val_str)

    ## Calculate MAPs in test dataset 
    fea, gt_label, pred_score = eval(test_loader, epoch, 'test')
    MAPs = np.zeros([n_view, n_view])
    test_dict = {}
    print_test_str = 'Test(mAP): '
    for i in range(n_view):
        for j in range(n_view):
            if i == j:
                continue
            MAPs[i, j] = utils.fx_calc_map_label(fea[j], gt_label[j], fea[i], gt_label[i], k=0, metric='cosine')[0]
            key = '%s2%s' % (args.views[i], args.views[j])
            test_dict[key] = MAPs[i, j]
            print_test_str = print_test_str + key + ': %g\t' % test_dict[key]

    test_avg = MAPs.sum() / n_view / (n_view - 1.)
    print_test_str = print_test_str + 'Avg: %g' % test_avg
    test_dict['avg'] = test_avg
    summary_writer.add_scalars('Retrieval/test', test_dict, epoch)

    if val_avg > best_acc:
        best_acc = val_avg
        print(print_test_str)
        print('Saving..')
        state = {}
        for v in range(n_view):
            state['model_state_dict_%d' % v] = multi_models[v].state_dict()
        for key in test_dict:
            state[key] = test_dict[key]
        state['epoch'] = epoch
        state['optimizer_state_dict'] = optimizer.state_dict()
        torch.save(state, os.path.join(args.ckpt_dir, '%s_%s_%d_best_checkpoint.t7' % ('MRL', args.data_name, args.output_dim)))
    return val_dict

def main():
    for epoch in range(start_epoch, args.max_epochs):
        train(epoch)
        lr_schedu.step(epoch)
        test_dict = test(epoch + 1)
        if test_dict['avg'] == best_acc:
            multi_model_state_dict = [{key: value.clone() for (key, value) in m.state_dict().items()} for m in multi_models]

    print('Evaluation on Last Epoch:')
    fea, lab, pred_score = eval(test_loader, epoch, 'test')
    test_dict, print_str = multiview_test(fea, lab)
    print(print_str)

    print('Evaluation on Best Validation:')
    [multi_models[v].load_state_dict(multi_model_state_dict[v]) for v in range(n_view)]
    fea, lab, pred_score = eval(test_loader, epoch, 'test')
    test_dict, print_str = multiview_test(fea, lab)
    print(print_str)

if __name__ == '__main__':
    best_acc = 0  # best test accuracy
    start_epoch = 0

    ## parsers
    args = step2_setting()
    args.log_dir = os.path.join(args.root_dir, 'logs', args.log_name)
    args.ckpt_dir = os.path.join(args.root_dir, 'ckpt', args.ckpt_dir)
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.ckpt_dir, exist_ok=True)
    print_log = open(args.log_file, 'w')
    
    print('===> Start..')
    ## GPU setting
    torch.cuda.manual_seed(args.seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    cudnn.benchmark = True

    # Dataset
    print('===> Preparing data ..')
    train_file_list = args.train_file_list
    train_dataset = cross_modal_dataset(args.data_name, noisy_mode=None, noisy_ratio=0, mode='train', 
                                        modal_list=args.views, image_file_list=train_file_list, image_transform=Image_transforms['train'],
                                        class_num=args.class_num)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        pin_memory=True,
        drop_last=False
    )
    test_file_list = args.test_file_list
    valid_dataset = cross_modal_dataset(args.data_name, noisy_mode=None, noisy_ratio=0, mode='valid', 
                                        modal_list=args.views, image_file_list=test_file_list, image_transform=Image_transforms['val'])
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        shuffle=False,
        drop_last=False
    )
    test_file_list = args.test_file_list
    test_dataset = cross_modal_dataset(args.data_name, noisy_mode=None, noisy_ratio=0, mode='test', 
                                       modal_list=args.views, image_file_list=test_file_list, image_transform=Image_transforms['test'])
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        shuffle=False,
        drop_last=False
    )

    ## Model
    print('===> Building Models..')
    multi_models = []
    n_view = len(args.views)

    for v in range(n_view):
        if v == args.views.index('RGBImg'):
            multi_models.append(models.__dict__['RGBImageNet'](args=args).cuda())
        elif v == args.views.index('PointCloud'):
            multi_models.append(models.__dict__['PointCloudNet'](args=args).cuda())
        else:
            multi_models.append(models.__dict__['RGBImageNet'](args=args).cuda())
    head_net = models.HeadNet(args.output_dim, args.class_num).cuda()

    rb_criterion = lossfunction.RCLLLoss(alpha=0.25, num_classes=args.class_num).cuda()
    cross_criterion = lossfunction.CrossModalLoss(tau=args.crossmodal_tau, num_classes=args.class_num, feat_dim=args.output_dim)
    
    parameters = []
    for v in range(n_view):
        parameters += list(multi_models[v].parameters())
    parameters += list(head_net.parameters())

    optimizer = torch.optim.SGD(parameters, lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    lr_schedu = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.max_epochs, eta_min=0, last_epoch=-1)
    summary_writer = SummaryWriter(args.log_dir)

    if args.resume:
        ckpt = torch.load(os.path.join(args.ckpt_dir, args.resume))
        for v in range(n_view):
            multi_models[v].load_state_dict(ckpt['model_state_dict_%d' % v])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch = ckpt['epoch']
        print('===> Load last checkpoint data')
    else:
        start_epoch = 0
        print('===> Start from scratch')

    ## main
    main()
    print_log.close()
