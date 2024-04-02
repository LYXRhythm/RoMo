# -*- coding=utf-8 -*-
import os
import numpy as np
import scipy.io as sio
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter

import nets as models
from paraser import step1_setting
import utils.utils as utils
import utils.losses as lossfunction
from utils.preprocess import *
from utils.bar_show import progress_bar
from utils.noisydataset import cross_modal_dataset
from utils.cluster import TKmeans, cluster_acc

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
        transforms.RandomCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224),
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

def generate_pseudo_labelling(data_loader, multi_model, file_list):
    gt_labels, pseudo_labels = [], []
    for batch_idx, (batches, targets, batches_path, index) in enumerate(data_loader):
        batches = [batches[v].cuda() for v in range(n_view)]
        
        multi_logits, multi_features = [], []
        for v in range(n_view):
            multi_result = multi_model[v](batches[v])
            multi_logits.append(multi_result[0].softmax(dim=1).argmax(axis=1))
            multi_features.append(multi_result[1])

        gt_labels.append(targets)
        pseudo_labels.append(multi_logits)

        for v in range(n_view):
            file_view = open(file_list[v], 'a', encoding='utf-8')
            for ii in range(len(batches_path[v])):
                write_temp = batches_path[v][ii]+" "+str(int(targets[v][ii]))+"\n"
                file_view.write(write_temp)
            file_view.close()

    for item1 in range(len(gt_labels)):
        for item2 in range(len(gt_labels[item1])):
            gt_labels[item1][item2] = torch.tensor(gt_labels[item1][item2].cpu().detach().numpy())
        gt_labels[item1] = torch.tensor([item.cpu().detach().numpy() for item in gt_labels[item1]])

    gt_labels = torch.tensor([item.cpu().detach().numpy() for item in gt_labels])
    gt_labels = gt_labels.transpose(0, 1)
    gt_labels = gt_labels.reshape(gt_labels.shape[0], -1)

    for item1 in range(len(pseudo_labels)):
        for item2 in range(len(pseudo_labels[item1])):
            pseudo_labels[item1][item2] = torch.tensor(pseudo_labels[item1][item2].cpu().detach().numpy())
        pseudo_labels[item1] = torch.tensor([item.cpu().detach().numpy() for item in pseudo_labels[item1]])
    pseudo_labels = torch.tensor([item.cpu().detach().numpy() for item in pseudo_labels])
    pseudo_labels = pseudo_labels.transpose(0, 1)
    pseudo_labels = pseudo_labels.reshape(pseudo_labels.shape[0], -1)

    return gt_labels.numpy(), pseudo_labels.numpy()

def train(epoch):
    print('\nEpoch: %d / %d' % (epoch, args.max_epochs), file=print_log)
    print('\nEpoch: %d / %d' % (epoch, args.max_epochs))
    train_loss, loss_list, total_list = 0., [0.] * n_view, [0.] * n_view

    ## Memory Bank Build
    set_eval()
    for v in range(n_view):
        if multi_models[v].classifier.memory_first:
            multi_models[v].feature_ext(train_loader=train_loader, v_th_views=v)

    ## Cluster Init
    features = []
    labels = []
    if epoch==0:
        features = [multi_models[v].classifier.memory for v in range(n_view)]
        labels = [multi_models[v].classifier.targets_memory for v in range(n_view)]

        features = torch.cat(features, dim=0).detach().cpu().numpy()
        labels = torch.cat(labels, dim=0).detach().cpu().numpy()
        ## First Step Cluster
        cluster_labels, cluster_centroids = TKmeans(features=features, k_classes=args.class_num, init_centroids=None)
        print('[PVI]: First Mixed Cluster ACC : {}'.format(cluster_acc(labels, cluster_labels)))
        ## Second Step Cluster
        cluster_labels_views = []
        cluster_centroids_views = []
        for v in range(n_view):
            feature_np = multi_models[v].classifier.memory.detach().cpu().numpy()
            cluster_result = TKmeans(features=feature_np, k_classes=args.class_num, init_centroids=cluster_centroids)
            cluster_labels_views.append(cluster_result[0])
            cluster_centroids_views.append(cluster_result[1])
            print('[PVI]: Second {} Modal Mixed Cluster ACC: {}'.format(args.views[v], cluster_acc(multi_models[v].classifier.targets_memory.detach().cpu().numpy(), cluster_labels_views[v])))

            with torch.no_grad():
                multi_models[v].self_top_layer.weight.copy_(torch.tensor(cluster_centroids_views[v]))
        print("\n\n")
    ## Training
    set_train()
    
    for batch_idx, (batches, targets, batches_path, index) in enumerate(train_loader):
        batches, targets, index = [batches[v].cuda() for v in range(n_view)], [targets[v].cuda() for v in range(n_view)], index.cuda()

        for v in range(n_view):
            multi_models[v].zero_grad()
        optimizer.zero_grad()
        
        multi_logits, multi_features = [], []
        for v in range(n_view):
            multi_result = multi_models[v](batches[v])
            multi_logits.append(multi_result[0])
            multi_features.append(multi_result[1])

        mem_logits = []
        for v in range(n_view):
            mem_logits.append(multi_models[v].self_top_layer(multi_models[v].classifier.memory[index]))

        # indomain
        indomain_losses = [in_criterion(multi_logits[v],
                            (torch.pow(mem_logits[v].softmax(dim=1), 0.9)/torch.sum(torch.pow(mem_logits[v].softmax(dim=1), 0.9))).argmax(axis=1)) for v in range(n_view)]
        indomain_loss = sum(indomain_losses)
        indomain_loss = 0.0

        ## crossdomain
        crossdomain_loss = cross_criterion(torch.cat((multi_features[0], multi_features[1]), dim=0))
        all_loss = args.lambda1 * indomain_loss + (1. - args.lambda1) * crossdomain_loss
        
        if epoch >= 0:
            all_loss.backward()
            optimizer.step()
        train_loss += all_loss.item()

        start_time = time.time()
        for v in range(n_view):
            multi_models[v].classifier.update_weight(multi_features[v], index)
        end_time = time.time()
        print("Epoch ", epoch, ": Training Time: ", end_time - start_time)

        for v in range(n_view):
            loss_list[v] += indomain_losses[v]
            _, predicted = multi_logits[v].max(1)
            total_list[v] += targets[v].size(0)
            acc = predicted.eq(targets[v]).sum().item()
        progress_bar(batch_idx, len(train_loader), 'Loss: %.3f | LR: %g' % (train_loss / (batch_idx + 1), optimizer.param_groups[0]['lr']))
        print('Loss: %.3f | LR: %g' % (train_loss / (batch_idx + 1), optimizer.param_groups[0]['lr']), file=print_log)
    

    train_dict = {('view_%d_loss' % v): loss_list[v] / len(train_loader) for v in range(n_view)}
    train_dict['sum_loss'] = train_loss / len(train_loader)
    summary_writer.add_scalars('Loss/train', train_dict, epoch)

    for v in range(n_view):
        multi_models[v].classifier.memory_first=False

def eval(data_loader, epoch, mode='test'):
    fea, lab, sample_path = [[] for _ in range(n_view)], [[] for _ in range(n_view)], [[] for _ in range(n_view)]
    test_loss, loss_list, total_list = 0., [0.] * n_view, [0.] * n_view
    with torch.no_grad():
        if sum([data_loader.dataset.train_data[v].shape[0] != data_loader.dataset.train_data[0].shape[0] for v in range(len(data_loader.dataset.train_data))]) == 0:
            for batch_idx, (batches, targets, batches_path, index) in enumerate(data_loader):
                batches, targets, index = [batches[v].cuda() for v in range(n_view)], [targets[v].cuda() for v in range(n_view)], index.cuda()

                multi_logits, multi_features = [], []
                for v in range(n_view):
                    multi_result = multi_models[v](batches[v])
                    multi_logits.append(multi_result[0])
                    multi_features.append(multi_result[1])
                
                pred, losses = [], []
                for v in range(n_view):
                    fea[v].append(multi_features[v])
                    lab[v].append(targets[v])
                    sample_path[v].append(batches_path[v])
                    pred.append(multi_logits[v])
                    losses.append(in_criterion(pred[v], targets[v]))
                    loss_list[v] += losses[v]
                    _, predicted = pred[v].max(1)
                    total_list[v] += targets[v].size(0)
                    acc = predicted.eq(targets[v]).sum().item()
                loss = sum(losses)
                test_loss += loss.item()
        else:
            pred, losses = [], []
            for v in range(n_view):
                count = int(np.ceil(data_loader.dataset.train_data[v].shape[0]) / data_loader.batch_size)
                for ct in range(count):
                    batch, targets = torch.Tensor(data_loader.dataset.train_data[v][ct * data_loader.batch_size: (ct + 1) * data_loader.batch_size]).cuda(), torch.Tensor(data_loader.dataset.noise_label[v][ct * data_loader.batch_size: (ct + 1) * data_loader.batch_size]).long().cuda()
                    
                    multi_logits, multi_features = multi_models[v](batches[v])
                    fea[v].append(multi_features)
                    lab[v].append(targets)
                    sample_path[v].append(batches_path[v])
                    pred.append(multi_logits)
                    losses.append(in_criterion(pred[v], targets))
                    loss_list[v] += losses[v]
                    _, predicted = pred[v].max(1)
                    total_list[v] += targets.size(0)
                    acc = predicted.eq(targets).sum().item()
                loss = sum(losses)
                test_loss += loss.item()

        fea = [torch.cat(fea[v]).cpu().detach().numpy() for v in range(n_view)]
        lab = [torch.cat(lab[v]).cpu().detach().numpy() for v in range(n_view)]
        sample_path_temp = []
        for v in range(n_view):
            for i in range(len(sample_path[v])):
                sample_path_temp = sample_path_temp + sample_path[v][i]
            sample_path[v] = sample_path_temp
            sample_path_temp = []
    test_dict = {('view_%d_loss' % v): loss_list[v] / len(data_loader) for v in range(n_view)}
    test_dict['sum_loss'] = test_loss / len(data_loader)
    summary_writer.add_scalars('Loss/' + mode, test_dict, epoch)

    return fea, lab, sample_path

def test(epoch):
    global best_acc
    global best_cluster_acc_score
    set_eval()
    
    fea, lab, sample_path = eval(train_loader, epoch, 'valid')

    paeudo_label = np.zeros([n_view, len(lab[0])]) - 1
    val_dict = {}

    for i in range(n_view):
        for j in range(n_view):
            if i == j:
                continue
            paeudo_label[i] = utils.get_label(fea[j], lab[j], fea[i], lab[i], k=0, metric='cosine')
    
    gt_labels, pseudo_labels = generate_pseudo_labelling(data_loader=train_loader, multi_model=multi_models, file_list=args.train_file_list_pseudo_labelling)
    cluster_acc_score1, cluster_acc_score2 = cluster_acc(lab[0], paeudo_label[0]), cluster_acc(lab[1], paeudo_label[1])
    cluster_acc_score = (cluster_acc_score1+cluster_acc_score2)/2.0
    print('[PVI]: After Training {} Modal Cluster ACC: {}'.format(args.views[0], cluster_acc_score1))
    print('[PVI]: After Training {} Modal Cluster ACC: {}'.format(args.views[1], cluster_acc_score2))

    print("len(sample_path[v]): ", len(sample_path[0]))

    if cluster_acc_score > best_cluster_acc_score: 
        best_cluster_acc_score = cluster_acc_score
        for v in range(len(args.train_file_list_pseudo_labelling)):
            os.remove(args.train_file_list_pseudo_labelling[v])
            with open(args.train_file_list_pseudo_labelling[v], "w") as f:
                for ii in range(len(sample_path[v])):
                    f.write(sample_path[v][ii]+" "+str(int(paeudo_label[v][ii]))+" "+str(int(lab[v][ii]))+"\n") 
    return val_dict

def main():
    for epoch in range(start_epoch, args.max_epochs):
        train(epoch)
        lr_schedu.step(epoch)
        test_dict = test(epoch + 1)
        multi_model_state_dict = [{key: value.clone() for (key, value) in m.state_dict().items()} for m in multi_models]

    fea, lab = eval(test_loader, epoch, 'test')
    test_dict, print_str = multiview_test(fea, lab)

    [multi_models[v].load_state_dict(multi_model_state_dict[v]) for v in range(n_view)]
    fea, lab = eval(test_loader, epoch, 'test')
    test_dict, print_str = multiview_test(fea, lab)
    
if __name__ == '__main__':
    best_acc = 0
    best_cluster_acc = 0
    best_cluster_acc_score = 0

    ## parsers
    args = step1_setting()
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
            multi_models.append(models.__dict__['UnsupervisedRGBImageNet'](args=args, memorySize=train_dataset.__len__()).cuda())
        elif v == args.views.index('PointCloud'):
            multi_models.append(models.__dict__['UnsupervisedPointCloudNet'](args=args, memorySize=train_dataset.__len__()).cuda())
        else:                            
            multi_models.append(models.__dict__['UnsupervisedRGBImageNet'](args=args, memorySize=train_dataset.__len__()).cuda())

    parameters = []
    for v in range(n_view):
        parameters += list(multi_models[v].parameters())

    optimizer = torch.optim.SGD(parameters, lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    lr_schedu = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.max_epochs, eta_min=0, last_epoch=-1)

    in_criterion = torch.nn.CrossEntropyLoss().cuda()
    cross_criterion = lossfunction.CrossModalLoss(tau=args.crossmodal_tau, num_classes=args.class_num, feat_dim=args.output_dim)
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
    
    ## dataset postprocessing    
    for v in range(len(args.train_file_list_pseudo_labelling)):
        filename = args.train_file_list_pseudo_labelling[v]
        backup_filename = filename+".backup"

        import shutil
        shutil.copyfile(filename, backup_filename)

        with open(filename, 'r') as f:
            lines = f.readlines()

        with open(filename, 'w') as f:
            f.writelines(lines[:train_dataset.__len__()])