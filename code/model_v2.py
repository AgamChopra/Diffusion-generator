#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 08:11:48 2024

@author: agam
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from math import ceil


def pad2d(inpt, target):  # pads if target is bigger and crops if smaller
    if torch.is_tensor(target):
        delta = [target.shape[2+i] - inpt.shape[2+i] for i in range(2)]
    else:
        delta = [target[i] - inpt.shape[2+i] for i in range(2)]
    output = nn.functional.pad(input=inpt,
                               pad=(ceil(delta[1]/2),
                                    delta[1] - ceil(delta[1]/2),
                                    ceil(delta[0]/2),
                                    delta[0] - ceil(delta[0]/2)),
                               mode='constant',
                               value=0).to(dtype=torch.float,
                                           device=inpt.device)
    return output


class Block(nn.Module):
    def __init__(self, in_c, embd_dim, out_c, hid_c=None, num_groups=8):
        super(Block, self).__init__()

        if hid_c is None:
            self.mlp = nn.Sequential(nn.Linear(embd_dim, out_c), nn.Mish())

            self.layer = nn.Sequential(
                MultiKernelConv2d(in_channels=in_c, out_channels=out_c),
                nn.Mish(),
                nn.GroupNorm(num_groups, out_c)
            )

            self.out_block = nn.Sequential(
                MultiKernelConv2d(in_channels=out_c, out_channels=out_c),
                nn.Mish(),
                nn.GroupNorm(num_groups, out_c)
            )
        else:
            self.mlp = nn.Sequential(nn.Linear(embd_dim, hid_c), nn.Mish())

            self.layer = nn.Sequential(
                MultiKernelConv2d(in_channels=in_c, out_channels=hid_c),
                nn.Mish(),
                nn.GroupNorm(num_groups, hid_c)
            )

            self.out_block = nn.Sequential(
                MultiKernelConv2d(in_channels=hid_c, out_channels=hid_c),
                nn.Mish(),
                nn.GroupNorm(num_groups, hid_c),
                nn.Upsample(scale_factor=2, mode='bilinear',
                            align_corners=True),
                MultiKernelConv2d(in_channels=hid_c, out_channels=out_c),
                nn.Mish(),
                nn.GroupNorm(num_groups, out_c)
            )

    def forward(self, x, t):
        t = self.mlp(t)
        y = self.layer(x)
        t = t[(..., ) + (None, ) * 2]
        y = y + t
        y = self.out_block(y)
        return y


class UNet(nn.Module):
    def __init__(self, CH=256, emb=64, n=4, num_groups=4):
        super(UNet, self).__init__()

        self.time_mlp = nn.Sequential(nn.Linear(emb, emb), nn.Mish())

        self.layer1 = nn.Sequential(
            MultiKernelConv2d(CH, int(128 * n)),
            nn.Mish(),
            nn.GroupNorm(num_groups, int(128 * n))
        )

        self.layer2 = Block(in_c=int(128 * n), embd_dim=emb,
                            out_c=int(256 * n), num_groups=num_groups)

        # self.layer3 = Block(in_c=int(128 * n), embd_dim=emb,
        #                     out_c=int(256 * n), num_groups=num_groups)

        # self.layer4 = Block(in_c=int(256 * n), embd_dim=emb,
        #                     out_c=int(512 * n), num_groups=num_groups)

        self.layer5 = Block(in_c=int(256 * n), embd_dim=emb,
                            out_c=int(256 * n), hid_c=int(512 * n), num_groups=num_groups)

        # self.layer6 = Block(in_c=int(1024 * n), embd_dim=emb,
        #                     out_c=int(256 * n), hid_c=int(512 * n), num_groups=num_groups)

        # self.layer7 = Block(in_c=int(512 * n), embd_dim=emb,
        #                     out_c=int(128 * n), hid_c=int(256 * n), num_groups=num_groups)

        self.layer8 = Block(in_c=int(512 * n), embd_dim=emb,
                            out_c=int(128 * n), num_groups=num_groups)

        self.out = nn.Sequential(
            nn.Conv2d(in_channels=int(128 * n), out_channels=CH, kernel_size=1)
        )

        self.pool2 = nn.Sequential(
            nn.Conv2d(in_channels=int(256 * n),
                      out_channels=int(256 * n), kernel_size=2, stride=2),
            nn.Mish(),
            nn.GroupNorm(num_groups, int(256 * n))
        )

        # self.pool3 = nn.Sequential(
        #     nn.Conv2d(in_channels=int(256 * n),
        #               out_channels=int(256 * n), kernel_size=2, stride=2),
        #     nn.Mish(),
        #     nn.GroupNorm(num_groups, int(256 * n))
        # )

        # self.pool4 = nn.Sequential(
        #     nn.Conv2d(in_channels=int(512 * n),
        #               out_channels=int(512 * n), kernel_size=2, stride=2),
        #     nn.Mish(),
        #     nn.GroupNorm(num_groups, int(512 * n))
        # )

    def forward(self, x, t):
        t = self.time_mlp(t)

        y = self.layer1(x)

        y2 = self.layer2(y, t)
        y = self.pool2(y2)

        # y3 = self.layer3(y, t)
        # y = self.pool3(y3)

        # y4 = self.layer4(y, t)
        # y = self.pool4(y4)

        y = self.layer5(y, t)

        # y = torch.cat((y4, pad2d(y, y4)), dim=1)
        # y = self.layer6(y, t)

        # y = torch.cat((y3, pad2d(y, y3)), dim=1)
        # y = self.layer7(y, t)

        y = torch.cat((y2, pad2d(y, y2)), dim=1)
        y = self.layer8(y, t)

        y = pad2d(y, x)

        y = self.out(y)

        return y


class Encoder(nn.Module):
    def __init__(self, CH=1, latent=256, num_groups=4):
        super(Encoder, self).__init__()

        # Add 1x1 convolutions to match the channel dimensions for residuals
        self.res_conv1 = nn.Conv2d(CH, int(latent/4), kernel_size=1)
        self.res_conv2 = nn.Conv2d(int(latent/4), int(latent/2), kernel_size=1)
        self.res_conv3 = nn.Conv2d(int(latent/2), latent, kernel_size=1)

        self.layer1 = nn.Sequential(
            MultiKernelConv2d(in_channels=CH, out_channels=int(latent/4)),
            nn.Mish(),
            nn.GroupNorm(num_groups, int(latent/4)),
            nn.Conv2d(in_channels=int(latent/4),
                      out_channels=int(latent/4), kernel_size=2, stride=2),
            nn.Mish(),
            nn.GroupNorm(num_groups, int(latent/4))
        )

        self.layer2 = nn.Sequential(
            MultiKernelConv2d(in_channels=int(latent/4),
                              out_channels=int(latent/2)),
            nn.Mish(),
            nn.GroupNorm(num_groups, int(latent/2)),
            nn.Conv2d(in_channels=int(latent/2),
                      out_channels=int(latent/2), kernel_size=2, stride=2),
            nn.Mish(),
            nn.GroupNorm(num_groups, int(latent/2))
        )

        self.layer3 = nn.Sequential(
            MultiKernelConv2d(in_channels=int(latent/2), out_channels=latent),
            nn.Mish(),
            nn.GroupNorm(num_groups, latent),
            nn.Conv2d(in_channels=latent, out_channels=latent,
                      kernel_size=2, stride=2),
            nn.Mish(),
            nn.GroupNorm(num_groups, latent)
        )

        # Linear layers for mean and log-variance
        self.fc_mu = nn.Conv2d(latent, latent, 1)
        self.fc_logvar = nn.Conv2d(latent, latent, 1)

    def forward(self, x):
        # Layer 1 with residual connection
        residual = self.res_conv1(x)  # Adjust channels
        y = self.layer1(x)
        y += nn.functional.interpolate(residual,
                                       size=y.shape[2:], mode='bilinear')

        # Layer 2 with residual connection
        residual = self.res_conv2(y)  # Adjust channels
        y = self.layer2(y)
        y += nn.functional.interpolate(residual,
                                       size=y.shape[2:], mode='bilinear')

        # Layer 3 with residual connection
        residual = self.res_conv3(y)  # Adjust channels
        y = self.layer3(y)
        y += nn.functional.interpolate(residual,
                                       size=y.shape[2:], mode='bilinear')

        # Calculate mu and logvar
        mu = self.fc_mu(y)
        logvar = self.fc_logvar(y)

        return mu, logvar


def reparameterize(mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std


class Decoder(nn.Module):
    def __init__(self, CH=1, latent=256, num_groups=4):
        super(Decoder, self).__init__()

        # Add 1x1 convolutions to match the channel dimensions for residuals
        self.res_conv1 = nn.Conv2d(latent, int(latent/2), kernel_size=1)
        self.res_conv2 = nn.Conv2d(int(latent/2), int(latent/4), kernel_size=1)
        self.res_conv3 = nn.Conv2d(int(latent/4), int(latent/8), kernel_size=1)

        self.layer1 = nn.Sequential(
            MultiKernelConv2d(in_channels=latent, out_channels=latent),
            nn.Mish(),
            nn.GroupNorm(num_groups, latent),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            MultiKernelConv2d(in_channels=latent, out_channels=int(latent/2)),
            nn.Mish(),
            nn.GroupNorm(num_groups, int(latent/2))
        )

        self.layer2 = nn.Sequential(
            MultiKernelConv2d(in_channels=int(latent/2),
                              out_channels=int(latent/2)),
            nn.Mish(),
            nn.GroupNorm(num_groups, int(latent/2)),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            MultiKernelConv2d(in_channels=int(latent/2),
                              out_channels=int(latent/4)),
            nn.Mish(),
            nn.GroupNorm(num_groups, int(latent/4))
        )

        self.layer3 = nn.Sequential(
            MultiKernelConv2d(in_channels=int(latent/4),
                              out_channels=int(latent/4)),
            nn.Mish(),
            nn.GroupNorm(num_groups, int(latent/4)),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            MultiKernelConv2d(in_channels=int(latent/4),
                              out_channels=int(latent/8)),
            nn.Mish(),
            nn.GroupNorm(num_groups, int(latent/8)),
        )

        self.out = nn.Conv2d(in_channels=int(latent/8),
                             out_channels=CH, kernel_size=1)

    def forward(self, mu, logvar=None):
        if logvar is not None:
            z = reparameterize(mu, logvar)
        else:
            z = mu

        # Layer 1 with residual connection
        residual = self.res_conv1(z)  # Adjust channels
        y = self.layer1(z)
        y += nn.functional.interpolate(residual,
                                       size=y.shape[2:], mode='bilinear')

        # Layer 2 with residual connection
        residual = self.res_conv2(y)  # Adjust channels
        y = self.layer2(y)
        y += nn.functional.interpolate(residual,
                                       size=y.shape[2:], mode='bilinear')

        # Layer 3 with residual connection
        residual = self.res_conv3(y)  # Adjust channels
        y = self.layer3(y)
        y += nn.functional.interpolate(residual,
                                       size=y.shape[2:], mode='bilinear')

        y = self.out(y)

        return y


class MultiKernelConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_sizes=[1, 3, 5, 7],
                 stride=1, padding=0, groups=1, apply_spectral_norm=False):
        super(MultiKernelConv2d, self).__init__()
        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels // len(kernel_sizes),
                      kernel_size=k, stride=stride, padding=k // 2,
                      groups=groups)
            for k in kernel_sizes
        ])
        if apply_spectral_norm:
            self.convs = nn.ModuleList(
                [nn.utils.spectral_norm(conv) for conv in self.convs])
        self.padding = padding

    def forward(self, input_tensor):
        out = torch.cat([conv(input_tensor) for conv in self.convs], dim=1)
        if self.padding > 0:
            out = F.pad(out, (self.padding, self.padding,
                        self.padding, self.padding))
        return out


def test_latent(device='cpu'):
    batch = 1
    x = torch.randn((batch, 3, 512, 512), device=device)
    t = torch.randn((batch, 64), device=device)
    print(x.shape, t.shape)

    enc = Encoder(CH=3, latent=128).to(device)
    dec = Decoder(CH=3, latent=128).to(device)
    lat = UNet(CH=128, n=8, emb=64).to(device)
    print('models loaded')

    mu, logvar = enc(x)
    print('mu, logvar shape:', mu.shape, logvar.shape)

    x_p = dec(mu, logvar)
    print(x_p.shape)

    z_t_p = lat(torch.rand_like(mu, device=mu.device), t)
    print('zp shape:', z_t_p.shape)
    print('shape z matches z_t_p:', mu.shape == z_t_p.shape)

    from matplotlib import pyplot as plt
    from helper import norm

    plt.imshow(norm(x)[0].permute(1, 2, 0).detach().cpu().numpy())
    plt.show()

    plt.imshow(norm(x_p)[0].permute(1, 2, 0).detach().cpu().numpy())
    plt.show()


if __name__ == '__main__':
    test_latent('cuda')
