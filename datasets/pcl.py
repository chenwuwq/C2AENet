import os
import torch
import numpy as np
from torch.utils.data import Dataset
from tqdm.auto import tqdm


def split_part(data, split_num=3, max_distance=3):
    # data: (B,N,C)
    B, N, C = data.shape
    print(data.shape)
    print('Split point cloud to part...')
    max_p = np.max(data, axis=1)[:, np.newaxis, :]  # (B,N,C)->(B,1,C)
    min_p = np.min(data, axis=1)[:, np.newaxis, :]
    diff = max_p - min_p  # (B,C)
    print(max_p, min_p, diff)


class PointCloudDataset(Dataset):

    def __init__(self, root, dataset, split, resolution, transform=None):
        super().__init__()
        self.dataset = dataset
        if dataset == "PUNet":
            self.pcl_dir = os.path.join(root, dataset, 'pointclouds', split, resolution)
        elif dataset == "Kinectv1":
            self.pcl_dir = os.path.join(root, split, "original_xyz")
            self.pcl_noisy_dir = os.path.join(root, split, "noisy_xyz")
        self.transform = transform
        self.pointclouds = []
        self.pointclouds_noisy = []
        self.pointcloud_names = []
        for fn in tqdm(os.listdir(self.pcl_dir), desc='Loading'):
            if fn[-3:] != 'xyz':
                continue
            pcl_path = os.path.join(self.pcl_dir, fn)
            if not os.path.exists(pcl_path):
                raise FileNotFoundError('File not found: %s' % pcl_path)
            pcl = torch.FloatTensor(np.loadtxt(pcl_path, dtype=np.float32))
            self.pointclouds.append(pcl)
            self.pointcloud_names.append(fn[:-4])

        if dataset == "Kinectv1":
            for fn in tqdm(os.listdir(self.pcl_noisy_dir), desc='Loading'):
                if fn[-3:] != 'xyz':
                    continue
                pcl_path = os.path.join(self.pcl_noisy_dir, fn)
                if not os.path.exists(pcl_path):
                    raise FileNotFoundError('File not found: %s' % pcl_path)
                pcl = torch.FloatTensor(np.loadtxt(pcl_path, dtype=np.float32))
                self.pointclouds_noisy.append(pcl)

    def __len__(self):
        return len(self.pointclouds)

    def __getitem__(self, idx):
        data = {
            'pcl_clean': self.pointclouds[idx].clone(),
            'name': self.pointcloud_names[idx]
        }
        if self.dataset == "Kinectv1":
            data = {
                'pcl_clean': self.pointclouds[idx].clone(),
                'pcl_noisy': self.pointclouds_noisy[idx].clone(),
                'name': self.pointcloud_names[idx]
            }
        if self.transform is not None:
            data = self.transform(data)
        return data

