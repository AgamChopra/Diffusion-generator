# -*- coding: utf-8 -*-
"""
Created on Sat Sep 16 19:41:29 2023

@author: agama
"""

import torch
from helper import show_images
from tqdm import trange
import cv2
import numpy as np


torch.set_printoptions(precision=9)


def get_cats_avg():
    img = cv2.imread('R:/E/target.jpg',)
    x = cv2.resize(img, dsize=(512, 512), interpolation=cv2.INTER_CUBIC).T
    img = np.asarray([x[2], x[1], x[0]]).astype(dtype=np.float32)
    img = (img - img.min()) / (img.max() - img.min())
    img = (img - 0.5) * 6
    return img.reshape((1, 3, 512, 512))


def load_cats():
    cat_list = []
    for i in trange(16):
        img = cv2.imread(
            'R:/E/ML/Dog-Cat-GANs/Dataset/cat_hq/cat (%d).jpg' % (i+1))
        x = cv2.resize(img, dsize=(512, 512),
                       interpolation=cv2.INTER_CUBIC).T
        cat_list.append([x[2], x[1], x[0]])
    print('.cat data loaded')
    return cat_list


def cat_dataset():
    cat = load_cats()
    return np.asanyarray(cat)


def fetch_data():
    data = cat_dataset()
    data = torch.from_numpy(data).to(dtype=torch.float)
    data = (data - data.min()) / (data.max() - data.min())
    data = (data - 0.5) * 6
    return data


def beta_cos(steps, start, end):
    x = torch.linspace(start, 1., steps, dtype=torch.float)
    beta = end * (1 - torch.cos(x * torch.pi / 2))
    return beta


def forward_sample(x0, t, steps, start=0.0001, end=0.02, scheduler='cos'):
    if scheduler == 'cos':
        betas = beta_cos(steps, start, end).to(x0.device)
    else:
        betas = torch.linspace(start, end, steps).to(x0.device)
    alphas = 1 - betas
    alpha_hat = torch.cumprod(alphas, dim=0)
    alpha_hat_t = torch.gather(alpha_hat, dim=-1,
                               index=t.to(x0.device)).view(-1, 1, 1, 1)
    noise = torch.randn_like(x0, device=x0.device)
    mean = alpha_hat_t.sqrt() * x0
    var = torch.sqrt(1 - alpha_hat_t) * noise
    xt = mean + var
    return xt, noise


steps = 1000

x = fetch_data()
print(x.shape)

tpop = range(1, steps)
idx = torch.linspace(0, 999, 16, dtype=torch.int)
imgs = []

for i in trange(0, steps):
    t = i * torch.ones((len(x)), dtype=torch.int64)
    xt, _ = forward_sample(x, t, steps, scheduler='lin')
    if i in idx:
        show_images(xt, 16, 4, mode=False)
        imgs.append(xt[6])
imgs = torch.stack(imgs, dim=0)
show_images(imgs.cpu(), 16, 4, mode=False)
