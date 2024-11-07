"""
Created on Wed Aug 23 19:21:10 2023

@author: Agam Chopra
"""

import os
import random
import torch
import torch.nn as nn
import torch.amp as amp
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import torch.optim.lr_scheduler as lr_scheduler
import matplotlib.pyplot as plt
from tqdm import tqdm, trange
import pandas as pd
from PIL import Image

from model_v2 import Encoder, Decoder, UNet, reparameterize
from helper import show_images, getPositionEncoding, forward_sample, Diffusion
from helper import distributions, norm  # , plot_grad_hist
from helper import KL_Loss, ssim_loss


print('cuda detected:', torch.cuda.is_available())
torch.set_printoptions(precision=9)

# 'highest', 'high', 'medium'. 'highest' is slower but accurate while 'medium'
#  is faster but less accurate. 'high' is preferred setting. Refer:
#  https://pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html
torch.set_float32_matmul_precision('medium')

# 'True' = faster but less accurate, 'False' = Slower but more accurate
#  has to be set to True if precision is high or medium
torch.backends.cuda.matmul.allow_tf32 = True

# 'True' = faster but less accurate, 'False' = Slower but more accurate
#  has to be set to True if precision is high or medium
torch.backends.cudnn.allow_tf32 = True

# For stability 'False', 'True', might be slower
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False


class CelebADataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.img_dir = os.path.join(root_dir, 'img_align_celeba')
        self.attr_file = os.path.join(root_dir, 'list_attr_celeba.txt')
        self.attrs = pd.read_csv(
            self.attr_file, delim_whitespace=True, header=1)
        self.img_names = self.attrs.index.tolist()

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_name = os.path.join(self.img_dir, self.img_names[idx])
        image = Image.open(img_name)
        if self.transform:
            image = self.transform(image)
        attrs = self.attrs.iloc[idx].values.astype('int')
        return image, attrs


class Loader:
    def __init__(self, batch_size=64, img_size=64, num_workers=8):

        print('loading data...')
        transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        dataset = CelebADataset(
            '/home/agam/Documents/git_projects/pytorch_datasets/celeba',
            transform=transform)
        self.data_loader = DataLoader(
            dataset=dataset, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, drop_last=True)
        self.iters = int(len(dataset) / batch_size)
        print('done.')


@torch.no_grad()
def fin(path, iterations=100, steps=1000, step=500):
    with torch.no_grad():
        data = Loader(batch_size=itr, img_size=img_size)
        enc = Encoder(CH=3, latent=128, num_groups=8).cuda()
        enc.load_state_dict(torch.load(os.path.join(path, "CELEB-enc.pt"),
                                       map_location='cuda'))
        enc.eval()

        dec = Decoder(CH=3, latent=128, num_groups=8).cuda()
        dec.load_state_dict(torch.load(os.path.join(path, "CELEB-dec.pt"),
                                       map_location='cuda'))
        dec.eval()

        for x in data.data_loader:
            x = norm(x[0]).cuda()
            mu, logvar = enc(x)
            z = reparameterize(mu, logvar)
            x_ae = dec(mu, logvar)
            break

        diffusion = Diffusion(steps=steps, scheduler='lin')

        model = UNet(CH=128, emb=64, n=8, num_groups=4).cuda()
        model.load_state_dict(
            torch.load(os.path.join(path, "CELEB-Autosave-Diffusion.pt"),
                       map_location='cuda'))
        model.eval()

        # z = torch.randn((iterations, 128, 8, 8)).cuda()
        z, _ = forward_sample(z, torch.tensor(
            steps - step).cuda(), steps, 'lin')
        z_min, z_max = z.min().item(), z.max().item()
        print(z_min, z_max)

        for t in trange(step, steps):
            z = torch.clamp(diffusion.backward(
                z, torch.tensor(steps - 1 - t), model), -5.5, 5.5)

        x_ldm = dec(z)

        return x.cpu(), x_ae.cpu(), x_ldm.cpu()


if __name__ == '__main__':
    path = '/home/agam/Documents/git_projects/'
    os.chdir(path)
    print(path)

    itr = 4
    img_size = 64
    steps = 1000
    start = 875

    with torch.no_grad():
        x, x_ae, x_ldm = fin(path, iterations=itr, steps=steps, step=start)

        show_images(torch.cat((norm(x[:itr]).cpu().detach(),
                               norm(x_ae[0:itr]).cpu().detach(),
                               norm(x_ldm[0:itr]).cpu().detach()), dim=0),
                    num_samples=int(itr*3), cols=itr,
                    dpi=250, size=(itr-1, 3))
