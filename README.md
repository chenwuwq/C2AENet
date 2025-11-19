# C2AENet
Official repo for "Progressive Point Cloud Denoising with Cross-Stage Cross-Coder Adaptive Edge Graph Convolution Network".

## Environment
```
conda create -n env_name python=3.9
conda activate env_name
conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=11.8 -c pytorch -c nvidia
conda install -c fvcore -c iopath -c conda-forge fvcore iopath
conda install -c bottler nvidiacub
conda install pytorch3d -c pytorch3d
conda install pyg -c pyg
```

## Train
You can simply train model with the following command:
```
python train.py
```
You can download the PUNet dataset [here]([https://github.com/luost26/score-denoise](https://github.com/ddsediri/IterativePFN/tree/main)).

## Test
You can simply test DHCN with the following command:
```
python test.py
```
The pre-trained checkpoint 'weight/ckpt.pt' can be readily used for evaluation.

## Citation
If you find our work useful, please give us star and cite our paper as:
```
inproceedings{chen2024progressive,
  title={Progressive Point Cloud Denoising with Cross-Stage Cross-Coder Adaptive Edge Graph Convolution Network},
  author={Chen, Wu and Fan, Hehe and Jiang, Qiuping and Huang, Chao and Yang, Yi},
  booktitle={Proceedings of the 32nd ACM International Conference on Multimedia},
  pages={6578--6587},
  year={2024}
}
```
