import os
import argparse
import torch
import torch.utils.tensorboard
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from tqdm.auto import tqdm

from datasets import *
from utils.misc import *
from utils.transforms import *
from utils.denoise import *
from models.denoise import *
from models.utils import chamfer_distance_unit_sphere


parser = argparse.ArgumentParser()
## Dataset and loader
# /media/cw/HDD1/denoise_dataset/PUNet/data,
parser.add_argument('--dataset_root', type=str, default='YOUR PATH/PUNet/data')
parser.add_argument('--dataset', type=str, default='PUNet', choices=['PUNet', 'Kinectv1'])
parser.add_argument('--patch_size', type=int, default=1000)
parser.add_argument('--num_patches', type=int, default=1000)
parser.add_argument('--resolutions', type=str_list, default=['10000_poisson', '30000_poisson', '50000_poisson'])
parser.add_argument('--noise_min', type=float, default=0.005)
parser.add_argument('--noise_max', type=float, default=0.020)
parser.add_argument('--train_batch_size', type=int, default=8)
# parser.add_argument('--val_batch_size', type=int, default=64)
parser.add_argument('--num_workers', type=int, default=16)
parser.add_argument('--aug_rotate', type=eval, default=True, choices=[True, False])
parser.add_argument('--supervised', type=eval, default=True, choices=[True, False])
## Optimizer and scheduler
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--weight_decay', type=float, default=0)
parser.add_argument('--max_grad_norm', type=float, default=float("inf"))
## Training
parser.add_argument('--seed', type=int, default=2020)
parser.add_argument('--logging', type=eval, default=False, choices=[True, False])
parser.add_argument('--log_root', type=str, default='./logs')
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--max_iters', type=int, default=100)
parser.add_argument('--val_noise', type=float, default=0.015)
parser.add_argument('--tag', type=str, default=None)
parser.add_argument('--resume', type=eval, default=False, choices=[True, False])
args = parser.parse_args()

seed = args.seed
seed_all(seed)

# Logging
if args.logging:
    log_dir = get_new_log_dir(args.log_root, prefix='D%s_' % (args.dataset),
                              postfix='_' + args.tag if args.tag is not None else '')
    logger = get_logger('train', log_dir)
    writer = torch.utils.tensorboard.SummaryWriter(log_dir)
    ckpt_mgr = CheckpointManager(log_dir)
    log_hyperparams(writer, log_dir, args)
else:
    logger = get_logger('train', None)
    ckpt_mgr = CheckpointManager("weight/")

logger.info(args)

# Datasets and loaders
logger.info('Loading datasets')
train_dset = PairedPatchDataset(
    datasets=[
        PointCloudDataset(
            root=args.dataset_root,
            dataset=args.dataset,
            split='train',
            resolution=resl,
            transform=standard_train_transforms(noise_std_max=args.noise_max, noise_std_min=args.noise_min,
                                                rotate=args.aug_rotate, add_noise=True)
        ) for resl in args.resolutions
    ],
    patch_size=args.patch_size,
    num_patches=args.num_patches,
    patch_ratio=1.2,
    on_the_fly=True
)
val_dset = PointCloudDataset(
    root=args.dataset_root,
    dataset=args.dataset,
    split='test',
    resolution=args.resolutions[0],
    transform=standard_train_transforms(noise_std_max=args.val_noise, noise_std_min=args.val_noise, rotate=False,
                                        scale_d=0, add_noise=True),
)
train_data = DataLoader(train_dset, batch_size=args.train_batch_size, num_workers=args.num_workers, shuffle=True)

# Modelyanz
logger.info('Building model...')
model = DenoiseNet(args).to(args.device)

if args.resume:
    ckpt = torch.load("weight path", map_location=args.device)
    model.load_state_dict(ckpt['state_dict'])

# Optimizer and scheduler
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=10, min_lr=1e-6)


# Train, validate and test
def train(it):
    # Load data
    loss_list = []
    for i, data in enumerate(tqdm(train_data, desc='Training')):
        pcl_noisy = data['pcl_noisy'].to(args.device)
        pcl_clean = data['pcl_clean'].to(args.device)
        pcl_seeds = data['seed_pnts'].to(args.device)
        pcl_std = torch.tensor(data['pcl_std']).to(args.device)

        # Reset grad and model state
        optimizer.zero_grad()
        model.train()

        # Forward
        loss = model.get_supervised_loss(
            pcl_noisy=pcl_noisy, pcl_clean=pcl_clean, pcl_seeds=pcl_seeds, pcl_std=pcl_std)

        loss_list.append(loss.item())
        # Backward and optimize
        loss.backward()
        optimizer.step()

    # Logging
    logger.info('[Train] Iter %04d | Loss %.6f' % (it, sum(loss_list) / len(loss_list)))


def validate(it):
    all_clean = []
    all_denoised = []
    for i, data in enumerate(tqdm(val_dset, desc='Validate')):
        pcl_noisy = data['pcl_noisy'].to(args.device)
        pcl_clean = data['pcl_clean'].to(args.device)
        pcl_denoised = patch_based_denoise(model, pcl_noisy, patch_size=args.patch_size, val_nostep=True)
        all_clean.append(pcl_clean.unsqueeze(0))
        all_denoised.append(pcl_denoised.unsqueeze(0))
    all_clean = torch.cat(all_clean, dim=0)
    all_denoised = torch.cat(all_denoised, dim=0)

    avg_chamfer = chamfer_distance_unit_sphere(all_denoised, all_clean, batch_reduction='mean')[0].item()
    logger.info('[Val] Iter %04d | CD %.4f  ' % (it, avg_chamfer * 100000))

    return avg_chamfer


# Main loop
logger.info('Start training...')
try:
    best_cd = 0.9999
    start_iter = 0
    if args.resume:
        start_iter = 0
    for it in range(1 + start_iter, args.max_iters + 1):
        lr = optimizer.param_groups[0]['lr']
        logger.info('iter:' + str(it) + ' current lr:' + str(lr))
        train(it)
        cd = validate(it)
        scheduler.step(cd)
        if cd < best_cd:
            best_cd = cd
            opt_states = {
                'optimizer': optimizer.state_dict(),
            }
            ckpt_mgr.save(model, args, best_cd, opt_states, step=it)

except KeyboardInterrupt:
    logger.info('Terminating...')
