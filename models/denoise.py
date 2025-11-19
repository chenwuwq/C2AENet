import torch
import pytorch3d.ops
import numpy as np
from .network import *


def get_random_indices(n, m):
    assert m < n
    # 将输入的数据进行随机排列
    return np.random.permutation(n)[:m]


class DenoiseNet(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.args = args

        self.num_modules = 4
        self.noise_decay = 4
        self.BCE_loss = nn.BCELoss()
        self.feature_net = C2AENet()

    def get_supervised_loss(self, pcl_noisy, pcl_clean, pcl_seeds, pcl_std):
        """
        Denoising score matching.
        Args:
            pcl_noisy:  Noisy point clouds, (B, N, 3).
            pcl_clean:  Clean point clouds, (B, M, 3). Usually, M is slightly greater than N.
        """
        B, N_noisy, N_clean, d = pcl_noisy.size(0), pcl_noisy.size(1), pcl_clean.size(1), pcl_noisy.size(2)
        pcl_seeds_1 = pcl_seeds.repeat(1, N_noisy, 1)

        pcl_noisy = pcl_noisy - pcl_seeds_1  # translate patch into origin
        pcl_seeds_2 = pcl_seeds.repeat(1, N_clean, 1)
        pcl_clean = pcl_clean - pcl_seeds_2

        loss_list = torch.zeros(self.num_modules)

        # network
        pcl_input = pcl_noisy
        curr_std = pcl_std
        pred_disp = self.feature_net(pcl_input)
        for i in range(self.num_modules):
            if i < self.num_modules // 2:
                curr_std = curr_std / self.noise_decay
                pcl_target = self.curr_iter_add_noise(pcl_clean, curr_std)
            else:
                pcl_target = pcl_clean

            _, _, clean_pts = pytorch3d.ops.knn_points(pcl_input, pcl_target, K=1, return_nn=True)
            clean_nbs = clean_pts.view(B, N_noisy, d)  # (B, N, 3)
            clean_nbs = clean_nbs - pcl_input
            pcl_input = pcl_input + pred_disp[i]

            dist = ((pred_disp[i] - clean_nbs) ** 2).sum(dim=-1)
            loss_list[i] = dist.sum(dim=-1).mean(dim=-1)

        final_loss = loss_list.sum()

        return final_loss, loss_list.sum()


    def denoise_langevin_dynamics(self, pcl_noisy):
        """
        Args:
            pcl_noisy:  Noisy point clouds, (B, N, 3).
        """
        with torch.no_grad():
            self.feature_net.eval()
            # Feature extraction
            pred_disp = self.feature_net(pcl_noisy)

        pred_clean = pcl_noisy
        for i in range(self.num_modules):
            pred_clean += pred_disp[i]

        return pred_clean

    def curr_iter_add_noise(self, pcl_clean, noise_std):
        new_pcl_clean = pcl_clean + torch.randn_like(pcl_clean) * noise_std.unsqueeze(1).unsqueeze(2)
        return new_pcl_clean.float()

