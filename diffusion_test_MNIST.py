#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 23 19:21:10 2023

@author: Agam Chopra
"""

import torch
import torchvision
import torch.nn as nn
import matplotlib.pyplot as plt
from math import ceil

# %% DATASET


def show_images(data, num_samples=9, cols=3):
    plt.figure(figsize=(5, 5))
    for i, img in enumerate(data):
        if i == num_samples:
            break
        plt.subplot(int(num_samples/cols) + 1, cols, i + 1)
        plt.axis('off')
        plt.imshow(img[0], cmap='gray_r')
    plt.show()


data = torchvision.datasets.MNIST(
    root='./data', train=True, download=True, transform=None).data
data = ((data[:, None] / 255) * 2) - 1
print(data.max(), data.min())
print(data.shape)

show_images(data)

# %% FORWARD STEP


def forward_sample(x0, t, steps, start=0., end=1.):
    betas = torch.linspace(start, end, steps)
    alphas = 1 - betas
    alpha_hat = torch.cumprod(alphas, dim=0)
    alpha_hat_t = torch.gather(alpha_hat, dim=-1, index=t).view(-1, 1, 1, 1)
    noise = torch.randn_like(x0) - 0.5
    mean = alpha_hat_t.sqrt() * x0
    var = torch.sqrt(1 - alpha_hat_t) * noise
    xt = mean + var
    return xt, noise


# %% TESTING FORWARD
T = 1000
x0 = torch.cat((data[0:1], data[0:1], data[0:1]), dim=0)
t = torch.tensor((999, 499, 0))
xt, noise = forward_sample(x0, t, steps=T, start=0.0001, end=0.02)
show_images(torch.cat((x0, noise, xt), dim=0))
print(noise.min(), noise.max(), noise.mean())
for a in xt:
    print(a.min(), a.max(), a.mean())
print(x0.min(), x0.max(), x0.mean())

# %% STEP/POSITIONAL EMBEDDINGS


def getPositionEncoding(seq_len, d=64, n=10000):
    P = torch.zeros((seq_len, d))
    for k in range(seq_len):
        for i in torch.arange(int(d/2)):
            denominator = torch.pow(n, 2*i/d)
            P[k, 2*i] = torch.sin(k/denominator)
            P[k, 2*i+1] = torch.cos(k/denominator)
    return P


# %% TESTING EMBEDDINGS

T = 1000
t = torch.tensor((999, 499, 0))
embds = getPositionEncoding(T, 64)
embds_t = embds[t]
plt.figure(figsize=(5, 10))
plt.imshow(embds)
plt.axis('off')
plt.show()
plt.figure(figsize=(10, 5))
plt.imshow(embds_t)
plt.axis('off')
plt.show()
print(embds.shape)
print(embds_t.shape)

# %% HELPER FUNCTIONS


def pad2d(input_, target):
    delta = [target.shape[2+i] - input_.shape[2+i] for i in range(2)]
    output = nn.functional.pad(input=input_,
                               pad=(ceil(delta[1]/2),
                                    delta[1] - ceil(delta[1]/2),
                                    ceil(delta[0]/2),
                                    delta[0] - ceil(delta[0]/2)),
                               mode='constant',
                               value=0).to(dtype=torch.float,
                                           device=input_.device)
    return output


# %% TESTING HELPER FUNCTIONS

a = torch.zeros(2, 3, 30, 31)
b = torch.ones(2, 3, 15, 17)
c = pad2d(b, a)
plt.imshow(a[0].T)
plt.axis('off')
plt.show()
plt.imshow(b[0].T)
plt.axis('off')
plt.show()
plt.imshow(c[0].T)
plt.axis('off')
plt.show()
print(a.shape, b.shape, c.shape)

# %% UNET


class Block(nn.Module):
    def __init__(self, in_c, embd_dim, out_c, hid_c=None):
        super(Block, self).__init__()

        if hid_c is None:
            self.mlp = nn.Sequential(nn.Linear(embd_dim, out_c), nn.ReLU())

            self.layer = nn.Sequential(nn.Conv2d(
                in_channels=in_c, out_channels=out_c, kernel_size=2), nn.ReLU(), nn.BatchNorm2d(out_c))

            self.out_block = nn.Sequential(nn.Conv2d(
                in_channels=out_c, out_channels=out_c, kernel_size=2), nn.ReLU(), nn.BatchNorm2d(out_c))
        else:
            self.mlp = nn.Sequential(nn.Linear(embd_dim, hid_c), nn.ReLU())

            self.layer = nn.Sequential(nn.Conv2d(
                in_channels=in_c, out_channels=hid_c, kernel_size=2), nn.ReLU(), nn.BatchNorm2d(hid_c))

            self.out_block = nn.Sequential(nn.Conv2d(in_channels=hid_c, out_channels=hid_c, kernel_size=2), nn.ReLU(), nn.BatchNorm2d(hid_c),
                                           nn.ConvTranspose2d(in_channels=hid_c, out_channels=out_c, kernel_size=2, stride=2), nn.ReLU(), nn.BatchNorm2d(out_c))

    def forward(self, x, t):
        t = self.mlp(t)
        y = self.layer(x)
        t = t[(..., ) + (None, ) * 2]
        y = y + t
        y = self.out_block(y)
        return y


class UNet(nn.Module):
    def __init__(self, CH=1, t_emb=64, n=1):
        super(UNet, self).__init__()
        # layers
        self.time_mlp = nn.Sequential(nn.Linear(t_emb, t_emb), nn.ReLU())

        self.layer1 = nn.Sequential(
            nn.Conv2d(CH, int(64/n), 2, 1), nn.ReLU(), nn.BatchNorm2d(int(64/n)))

        self.layer2 = Block(in_c=int(64/n), embd_dim=t_emb, out_c=int(128/n))

        self.layer3 = Block(in_c=int(128/n), embd_dim=t_emb,
                            out_c=int(128/n), hid_c=int(256/n))

        self.layer4 = Block(in_c=int(256/n), embd_dim=t_emb, out_c=int(64/n))

        self.out = nn.Sequential(nn.Conv2d(in_channels=int(64/n),
                                           out_channels=int(64/n),
                                           kernel_size=1),
                                 nn.ReLU(), nn.BatchNorm2d(int(64/n)),
                                 nn.Conv2d(in_channels=int(64/n),
                                           out_channels=CH, kernel_size=1))

        self.pool2 = nn.Conv2d(in_channels=int(
            128/n), out_channels=int(128/n), kernel_size=2, stride=2)

    def forward(self, x, t):
        t = self.time_mlp(t)
        y = self.layer1(x)

        y2 = self.layer2(y, t)
        y = self.pool2(y2)

        y = self.layer3(y, t)

        y = torch.cat((y2, pad2d(y, y2)), dim=1)
        y = self.layer4(y, t)

        y = pad2d(y, x)

        y = self.out(y)

        return y


# %% TESTING UNET
with torch.no_grad():
    a = xt
    embd = embds_t
    model = UNet()
    model.eval()
    b = model(a, embd)
    show_images(torch.cat((a, b), dim=0))
    print(a.shape, b.shape)

# %% DIFFUSION MODEL


class Diffusion:
    def __init__(self, start=0.0001, end=0.01, steps=1000):
        self.start = start
        self.end = end
        self.steps = steps
        self.beta = torch.linspace(start, end, steps)
        self.alpha = 1 - self.betas
        self.alpha_hat = torch.cumprod(self.alphas, dim=0)

    def forward(self, x0, t):
        noise = torch.randn_like(x0) - 0.5
        alpha_hat_t = torch.gather(self.alpha_hat, dim=-1,
                                   index=t).view(-1, 1, 1, 1)
        mean = alpha_hat_t.sqrt() * x0
        var = torch.sqrt(1 - alpha_hat_t) * noise
        xt = mean + var
        return xt, noise

    @torch.no_grad()
    def backward(self, x, t, model):
        beta_t = torch.gather(self.beta, dim=-1, index=t).view(-1, 1, 1, 1)
        sqrt_one_minus_alpha_hat_t = torch.gather(torch.sqrt(
            1. - self.alpha_hat), dim=-1, index=t).view(-1, 1, 1, 1)
        sqrt_inv_alpha_t = torch.gather(torch.sqrt(
            1.0 / self.alpha), dim=-1, index=t).view(-1, 1, 1, 1)
        mean = sqrt_inv_alpha_t * \
            (x - beta_t * model(x, t) / sqrt_one_minus_alpha_hat_t)
        posterior_variance_t = beta_t

        if t == 0:
            return mean
        else:
            noise = torch.randn_like(x)
            varience = torch.sqrt(posterior_variance_t) * noise
            return mean + varience


# %%
