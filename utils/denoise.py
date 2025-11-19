import math
import torch
import numpy as np
import pytorch3d.ops
from tqdm.auto import tqdm
from models.utils import farthest_point_sampling


def patch_based_denoise(model, pcl_noisy, patch_size=1000, seed_k=5, step_alpha=1, val_nostep=False):
    """
    Args:
        pcl_noisy:  Input point cloud, (N, 3)
    """
    assert pcl_noisy.dim() == 2, 'The shape of input point cloud must be (N, 3).'
    N, d = pcl_noisy.size()
    pcl_noisy = pcl_noisy.unsqueeze(0)  # (1, N, 3)
    patch_num = int(seed_k * N / patch_size)
    seed_pnts, _ = farthest_point_sampling(pcl_noisy, patch_num)
    patch_dists, point_idxs_in_main_pcd, patches = pytorch3d.ops.knn_points(seed_pnts, pcl_noisy, K=patch_size,
                                                                            return_nn=True)
    patches = patches[0]  # (N, K, 3)
    seed_pnts_1 = seed_pnts.squeeze().unsqueeze(1).repeat(1, patch_size, 1)
    patches = patches - seed_pnts_1
    patch_dists, point_idxs_in_main_pcd = patch_dists[0], point_idxs_in_main_pcd[0]
    patch_dists = patch_dists / patch_dists[:, -1].unsqueeze(1).repeat(1, patch_size)

    all_dists = torch.ones(patch_num, N) / 0
    all_dists = all_dists.cuda()
    all_dists = list(all_dists)
    patch_dists, point_idxs_in_main_pcd = list(patch_dists), list(point_idxs_in_main_pcd)

    for all_dist, patch_id, patch_dist in zip(all_dists, point_idxs_in_main_pcd, patch_dists):
        all_dist[patch_id] = patch_dist
    all_dists = torch.stack(all_dists, dim=0)
    weights = torch.exp(-1 * all_dists)
    best_weights, best_weights_idx = torch.max(weights, dim=0)

    if val_nostep:
        with torch.no_grad():
            model.eval()
            patches_denoised = model.denoise_langevin_dynamics(patches)

    else:
        i = 0
        patch_step = int(N / (step_alpha * patch_size))
        patches_denoised = []
        with torch.no_grad():
            model.eval()
            while i < patch_num:
                curr_patches = patches[i:i + patch_step]
                curr_denoised = model.denoise_langevin_dynamics(curr_patches)
                patches_denoised.append(curr_denoised)
                i += patch_step
        patches_denoised = torch.cat(patches_denoised, dim=0)

    patches_denoised += seed_pnts_1
    pcl_denoised = [patches_denoised[patch][point_idxs_in_main_pcd[patch] == pidx_in_main_pcd] for
                    pidx_in_main_pcd, patch in enumerate(best_weights_idx)]

    pcl_denoised = torch.cat(pcl_denoised, dim=0)
    return pcl_denoised


