import os
import time
import argparse
import torch
from datasets import *
from utils.misc import *
from utils.denoise import *
from utils.transforms import *
from utils.evaluate import *
from models.denoise import *


def input_iter(input_dir):
    for fn in os.listdir(input_dir):
        if fn[-3:] != 'xyz':
            continue
        pcl_noisy = torch.FloatTensor(np.loadtxt(os.path.join(input_dir, fn)))
        pcl_noisy, center, scale = NormalizeUnitSphere.normalize(pcl_noisy)
        yield {
            'pcl_noisy': pcl_noisy,
            'name': fn[:-4],
            'center': center,
            'scale': scale
        }


# Arguments
parser = argparse.ArgumentParser()
parser.add_argument('--ckpt', type=str, default='./weight/ckpt.pt')
parser.add_argument('--input_root', type=str, default='YOUR PATH/PUNet/data/examples')
parser.add_argument('--output_root', type=str, default='YOUR PATH/PUNet/data/results')
parser.add_argument('--dataset_root', type=str, default='YOUR PATH/PUNet/data')
parser.add_argument('--clean_root', type=str,
                    default='YOUR PATH/PUNet/data/PUNet/pointclouds/test')
parser.add_argument('--dataset', type=str, default='PUNet')
parser.add_argument('--tag', type=str, default='')
parser.add_argument('--resolution', type=str, default='10000_poisson')
parser.add_argument('--noise', type=str, default='0.025')
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--seed', type=int, default=2020)
# Denoiser parameters
parser.add_argument('--seed_k', type=int, default=6)
parser.add_argument('--niters', type=int, default=1)
args = parser.parse_args()
seed_all(args.seed)

# Input/Output
input_dir = os.path.join(args.input_root, '%s_%s_%s' % (args.dataset, args.resolution, args.noise))
print(input_dir)
save_title = '{dataset}_Ours{modeltag}_{tag}_{res}_{noise}_{time}'.format_map({
    'dataset': args.dataset,
    'modeltag': '' if args.niters == 1 else '%dx' % args.niters,
    'tag': args.tag,
    'res': args.resolution,
    'noise': args.noise,
    'time': time.strftime('%m-%d-%H-%M-%S', time.localtime())
})
output_dir = os.path.join(args.output_root, save_title)
os.makedirs(output_dir)
os.makedirs(os.path.join(output_dir, 'pcl'))  # Output point clouds
logger = get_logger('test', output_dir)
for k, v in vars(args).items():
    logger.info('[ARGS::%s] %s' % (k, repr(v)))

# Model
ckpt = torch.load(args.ckpt, map_location=args.device)
model = DenoiseNet(ckpt['args']).to(args.device)
model.load_state_dict(ckpt['state_dict'])

# Denoise
for data in input_iter(input_dir):
    logger.info(data['name'])
    pcl_noisy = data['pcl_noisy'].to(args.device)
    pcl_clean = torch.FloatTensor(np.loadtxt(os.path.join(args.clean_root, args.resolution, data['name'] + ".xyz"))).to(
        args.device)
    with torch.no_grad():
        model.eval()
        pcl_next = pcl_noisy
        for _ in range(args.niters):
            pcl_next = patch_based_denoise(model=model, pcl_noisy=pcl_next, patch_size=1000, seed_k=args.seed_k)
        pcl_denoised = pcl_next.cpu()
        # Denormalize
        pcl_denoised = pcl_denoised * data['scale'] + data['center']

    save_path = os.path.join(output_dir, 'pcl', data['name'] + '.xyz')
    np.savetxt(save_path, pcl_denoised.numpy(), fmt='%.8f')

# Evaluate
evaluator = Evaluator(
    output_pcl_dir=os.path.join(output_dir, 'pcl'),
    dataset_root=args.dataset_root,
    dataset=args.dataset,
    summary_dir=args.output_root,
    experiment_name=save_title,
    device=args.device,
    res_gts=args.resolution,
    logger=logger
)
evaluator.run()
