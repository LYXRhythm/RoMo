import argparse

def step1_setting():
    parser = argparse.ArgumentParser(description='RoMo: Robust Unsupervised Multimodal Learning for Cross-Modal Retrieval')
    # GPU
    parser.add_argument('--gpu_id', type=str, nargs='?', default='1', 
                    help="device id to run")
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
    # File Setting
    parser.add_argument('--root_dir', type=str, default='./')
    parser.add_argument('--log_file', type=str, default='./logs/stage1_current_log.txt')
    # Dataset
    parser.add_argument("--data_name", type=str, default="mnist3d", help="data name")
    parser.add_argument('--log_name', type=str, default='RoMo_PLA_Stage')
    parser.add_argument('--ckpt_dir', type=str, default='RoMo_PLA_Stage')
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--train_file_list', type=list, default=["./dataset/mnist3d_RGBImgs_train_list.txt", "./dataset/mnist3d_PointCloud_train_list.txt"]) 
    parser.add_argument('--test_file_list', type=list, default=["./dataset/mnist3d_RGBImgs_test_list.txt", "./dataset/mnist3d_PointCloud_test_list.txt"]) 
    parser.add_argument('--train_file_list_pseudo_labelling', type=list, default=["./mnist3d_RGBImgs_train_list_pseudo_labelling.txt", "./mnist3d_PointCloud_train_list_pseudo_labelling.txt"])
    parser.add_argument('--views', nargs='+', help='<Required> Quantization bits', default=['RGBImg', 'PointCloud']) #2D:{RGBImg, GrayImg} 3D:{PointCloud, Mesh}
    # Model
    parser.add_argument('--backbone_2d', type=str, default='resnet18', choices=['resnet18','resnet50','resnet101'],
                    help='which image backbone network')
    parser.add_argument('--backbone_3d', type=str, default='dgcnn', choices=['dgcnn','pointnet','mesh'],
                    help='which 3d backbone network ')
    parser.add_argument('--model3d_path', type=str, default='./pretrained/model.1024.t7',
                    help='3d backbone network path')
    parser.add_argument('--pretretrained', type=bool, default=True, help='pretretrained')
    parser.add_argument('--k', type=int, default=20, metavar='N', help='Num of nearest neighbors to use')
    parser.add_argument('--membank_t', default=0.1, type=float,
                        metavar='T', help='temperature parameter for softmax')
    parser.add_argument('--membank_m', default=0.9, type=float,
                        metavar='M', help='momentum for non-parametric updates')
    # Hyper-parameters
    parser.add_argument('--resume', default=None, type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
    parser.add_argument('--max_epochs', type=int, default=30, help='maximum epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--train_batch_size', type=int, default=50)
    parser.add_argument('--eval_batch_size', type=int, default=50)
    parser.add_argument('--output_dim', type=int, default=1024, help='output dimension')
    parser.add_argument('--class_num', type=int, default=10)
    parser.add_argument('--lambda1', type=float, default=0.75)
    parser.add_argument('--indomain_tau', type=float, default=0.1)
    parser.add_argument('--crossmodal_tau', type=float, default=0.05)
    args = parser.parse_args()
    return args

def step2_setting():
    parser = argparse.ArgumentParser(description='RoMo: Robust Unsupervised Multimodal Learning for Cross-Modal Retrieval')
    # GPU
    parser.add_argument('--gpu_id', type=str, nargs='?', default='1',
                    help="device id to run")
    parser.add_argument('--seed', type=int, default=0, metavar='S',
                    help='random seed (default: 1)')
    # File Setting
    parser.add_argument('--root_dir', type=str, default='./')
    parser.add_argument('--log_file', type=str, default='./logs/stage2_current_log.txt')
    # Dataset
    parser.add_argument("--data_name", type=str, default="mnist3d", help="data name")
    parser.add_argument('--log_name', type=str, default='RoMo_LNP_Stage')
    parser.add_argument('--ckpt_dir', type=str, default='RoMo_LNP_Stage')
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--train_file_list', type=list, default=["./mnist3d_RGBImgs_train_list_pseudo_labelling.txt", "./mnist3d_PointCloud_train_list_pseudo_labelling.txt"]) # ["./imgs_train_list.txt", "./points3d_train_list.txt"]
    parser.add_argument('--test_file_list', type=list, default=["./dataset/mnist3d_RGBImgs_test_list.txt", "./dataset/mnist3d_PointCloud_test_list.txt"])
    parser.add_argument('--views', nargs='+', help='<Required> Quantization bits', default=['RGBImg', 'PointCloud'])
    # Model
    parser.add_argument('--backbone_2d', type=str, default='resnet18', choices=['resnet18','resnet50','resnet101'],
                    help='which image network')
    parser.add_argument('--backbone_3d', type=str, default='dgcnn', choices=['dgcnn','pointnet','mesh'],
                    help='which 3d backbone network ')
    parser.add_argument('--model3d_path', type=str, default='./pretrained/model.1024.t7',
                    help='which 3d backbone pretrained model ')
    parser.add_argument('--pretretrained', type=bool, default=True, help='pretretrained')
    parser.add_argument('--k', type=int, default=40, metavar='N', help='Num of nearest neighbors to use')
    # Hyper-parameters
    parser.add_argument('--resume', default=None, type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
    parser.add_argument('--max_epochs', type=int, default=1)
    parser.add_argument('--lr', type=float, default=0.0002, help='learning rate') # 0.00001
    parser.add_argument('--weight_decay', type=float, default=1e-6)
    parser.add_argument('--train_batch_size', type=int, default=50) # 20
    parser.add_argument('--eval_batch_size', type=int, default=50)  # 20
    parser.add_argument('--output_dim', type=int, default=256, help='output dimension') 
    parser.add_argument('--class_num', type=int, default=10)
    parser.add_argument('--lambda_rb', type=float, default=0.7)       
    parser.add_argument('--lambda_crossmodal', type=float, default=0.3)
    parser.add_argument('--crossmodal_tau', type=float, default=0.2) # 0.05
    
    args = parser.parse_args()
    return args