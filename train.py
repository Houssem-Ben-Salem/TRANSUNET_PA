import argparse
import logging
import os
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from networks.vit_seg_modeling import VisionTransformer as ViT_seg
from networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
from trainer import trainer_parse_improved

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='../data/Synapse/train_npz', help='root dir for data')
parser.add_argument('--dataset', type=str,
                    default='Synapse', help='experiment_name')
parser.add_argument('--list_dir', type=str,
                    default='./lists/lists_Synapse', help='list dir')
parser.add_argument('--num_classes', type=int,
                    default=9, help='output channel of network')
parser.add_argument('--max_iterations', type=int,
                    default=30000, help='maximum epoch number to train')
parser.add_argument('--max_epochs', type=int,
                    default=150, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int,
                    default=24, help='batch_size per gpu')
parser.add_argument('--n_gpu', type=int, default=1, help='total gpu')
parser.add_argument('--deterministic', type=int,  default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float, default=0.0001,
                    help='segmentation network learning rate')
parser.add_argument('--img_size', type=int,
                    default=224, help='input patch size of network input')
parser.add_argument('--seed', type=int,
                    default=1234, help='random seed')
parser.add_argument('--n_skip', type=int,
                    default=3, help='using number of skip-connect, default is num')
parser.add_argument('--vit_name', type=str,
                    default='R50-ViT-B_16', help='select one vit model')
parser.add_argument('--vit_patches_size', type=int,
                    default=16, help='vit_patches_size, default is 16')
parser.add_argument('--val_frequency', type=int,
                    default=5000, help='validation frequency in iterations')
parser.add_argument('--resume', action='store_true',
                    help='resume training from checkpoint if exists')

# New arguments for the improved implementation
parser.add_argument('--improved', action='store_true',
                    help='use improved training pipeline')
parser.add_argument('--pa_slice_ratio', type=float, default=0.8,
                    help='ratio of positive PA slices in training (for improved pipeline)')
parser.add_argument('--patch_size', type=int, default=None,
                    help='size of patches to extract (None for whole slice, for improved pipeline)')
parser.add_argument('--tversky_beta', type=float, default=0.7,
                    help='beta parameter for Tversky loss (higher values prioritize recall)')
parser.add_argument('--ce_weight', type=float, default=0.3,
                    help='weight for CE loss component in combined loss')
parser.add_argument('--elastic_deform_prob', type=float, default=0.3,
                    help='probability of applying elastic deformation during augmentation')

args = parser.parse_args()


if __name__ == "__main__":
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    dataset_name = args.dataset
    # Add to the dataset_config dictionary
    dataset_config = {
        'Parse': {
            'root_path': './DATA',
            'list_dir': './lists/lists_Parse',
            'num_classes': 2,  # Background and PA
        },
    }
    args.num_classes = dataset_config[dataset_name]['num_classes']
    args.root_path = dataset_config[dataset_name]['root_path']
    args.list_dir = dataset_config[dataset_name]['list_dir']
    args.is_pretrain = True
    
    # Modify experiment name to indicate improved version if using it
    improved_suffix = '_improved' if args.improved else ''
    args.exp = 'TU_' + dataset_name + str(args.img_size) + improved_suffix
    
    snapshot_path = "../model/{}/{}".format(args.exp, 'TU')
    snapshot_path = snapshot_path + '_pretrain' if args.is_pretrain else snapshot_path
    snapshot_path += '_' + args.vit_name
    snapshot_path = snapshot_path + '_skip' + str(args.n_skip)
    snapshot_path = snapshot_path + '_vitpatch' + str(args.vit_patches_size) if args.vit_patches_size!=16 else snapshot_path
    snapshot_path = snapshot_path+'_'+str(args.max_iterations)[0:2]+'k' if args.max_iterations != 30000 else snapshot_path
    snapshot_path = snapshot_path + '_epo' +str(args.max_epochs) if args.max_epochs != 30 else snapshot_path
    snapshot_path = snapshot_path+'_bs'+str(args.batch_size)
    snapshot_path = snapshot_path + '_lr' + str(args.base_lr) if args.base_lr != 0.01 else snapshot_path
    snapshot_path = snapshot_path + '_'+str(args.img_size)
    snapshot_path = snapshot_path + '_s'+str(args.seed) if args.seed!=1234 else snapshot_path
    
    # Add additional path components for improved version
    if args.improved:
        if args.patch_size is not None:
            snapshot_path = snapshot_path + f'_patch{args.patch_size}'
        snapshot_path = snapshot_path + f'_paratio{args.pa_slice_ratio}'
        snapshot_path = snapshot_path + f'_tvbeta{args.tversky_beta}'

    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    config_vit = CONFIGS_ViT_seg[args.vit_name]
    config_vit.n_classes = args.num_classes
    config_vit.n_skip = args.n_skip
    if args.vit_name.find('R50') != -1:
        config_vit.patches.grid = (int(args.img_size / args.vit_patches_size), int(args.img_size / args.vit_patches_size))
    net = ViT_seg(config_vit, img_size=args.img_size, num_classes=config_vit.n_classes).cuda()
    
    # Only load from pretrained if not resuming from checkpoints
    if not args.resume:
        net.load_from(weights=np.load(config_vit.pretrained_path))
        print("Loaded pretrained weights")
    else:
        print("Will try to resume from checkpoint")

    # Define trainers, including the improved version
    trainers = {
        'Parse': trainer_parse_improved 
    }
    
    print(f"Using {'improved' if args.improved else 'original'} training pipeline for {dataset_name}")
    trainers[dataset_name](args, net, snapshot_path)