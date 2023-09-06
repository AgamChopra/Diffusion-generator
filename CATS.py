"""
Created on Wed Aug 23 19:21:10 2023

@author: Agam Chopra
"""

import os
import random
import numpy as np
import cv2
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm, trange
from torch.utils.data import DataLoader

from models import UNet
from helper import show_images, getPositionEncoding, forward_sample, Diffusion
from helper import distributions


def load_cats():
    cat_list = []
    for i in trange(5650):
        img = cv2.imread(
            'E:/ML/Dog-Cat-GANs/Dataset/cat_hq/cat (%d).jpg' % (i+1))
        cat_list.append(cv2.resize(img, dsize=(64, 64),
                        interpolation=cv2.INTER_CUBIC))
    print('.cat data loaded')
    return cat_list


def cat_dataset():
    cat = load_cats()
    return np.swapaxes(np.asanyarray(cat), 1, -1)


def fetch_data():
    data = cat_dataset()
    data = torch.from_numpy(data).to(dtype=torch.float)
    data = (data - data.min()) / (data.max() - data.min())
    return data


class Loader:
    def __init__(self, batch_size=32, num_workers=4):
        dataset = fetch_data()
        self.data_loader = DataLoader(
            dataset=dataset, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, drop_last=True)
        self.iters = int(len(dataset) / batch_size)


def train(path, epochs=2000, lr=1E-6, batch_size=64, steps=1000, n=1, emb=64,
          err_func=nn.L1Loss(), device='cpu'):
    data = Loader(batch_size=batch_size)
    model = UNet(CH=3, emb=emb, n=n).to(device)
    try:
        model.load_state_dict(torch.load(os.path.join(path,
                                                      "CATS-Autosave.pt")))
    except Exception:
        print('paramerts failed to load from last run')
    optimizer = torch.optim.AdamW(model.parameters(), lr)
    train_error = []
    avg_fact = data.iters
    tpop = range(1, steps)
    embds = getPositionEncoding(steps).to(device)

    for eps in range(epochs):
        print(f'Epoch {eps + 1}|{epochs}')
        model.train()
        running_error = 0.

        for itr, x in enumerate(tqdm(data.data_loader)):
            optimizer.zero_grad()
            x0 = x.to(device)
            t = torch.tensor(random.sample(tpop, len(x0)), device=device)
            xt, noise = forward_sample(x0, t, steps, 0.0001, 0.02)
            embdt = embds[t]

            pred_noise = model(xt, embdt)
            error = err_func(noise, pred_noise)
            error.backward()
            optimizer.step()

            running_error += error.item()

            if itr >= avg_fact:
                break

        train_error.append(running_error / avg_fact)
        print(f'Average Error: {train_error[-1]}')

        if eps % 10 == 0:
            plt.figure(figsize=(10, 5))
            plt.title("Training Noise Prediction Error")
            plt.plot(train_error, label='Average Error per Epoch')
            plt.legend()
            plt.xlabel("Epochs")
            plt.ylabel("Average Error")
            plt.legend()
            plt.show()

            torch.save(model.state_dict(),
                       os.path.join(path, "CATS-Autosave.pt"))


def fin(iterations=100):
    diffusion = Diffusion(steps=1000)

    path = 'R:/git projects/parameters/'
    model = UNet(CH=3, emb=64, n=0.5).cuda()
    model.load_state_dict(torch.load(os.path.join(path,
                                                  "CATS-Autosave.pt")))
    idx = torch.linspace(0, 999, 16, dtype=torch.int).cuda()
    print(idx)

    x = torch.randn((iterations, 3, 64, 64)).cuda()
    imgs = []

    for t in trange(0, 1000):
        x = torch.clamp(diffusion.backward(
            x, torch.tensor(999 - t), model), 0, 1)
        if t in idx:
            imgs.append(x[0:1])

    imgs = torch.cat(imgs, dim=0)
    show_images(imgs.cpu(), 16, 4)
    show_images(x.cpu(), iterations, int(iterations ** 0.5))
    return x.cpu()


if __name__ == '__main__':
    itr = 64
    a = input('Train model from last checkpoint?(y/n)')
    if a == 'y':
        train(path='R:/git projects/parameters/', epochs=2000,
              err_func=nn.HuberLoss(delta=0.01), lr=1E-4, batch_size=64,
              steps=1000, n=0.5, emb=64, device='cuda')
    else:
        y = fin(iterations=itr)
        data = Loader(batch_size=itr)
        for x in data.data_loader:
            break
        distributions(x[:, 0], y[:, 0])
        distributions(x[:, 1], y[:, 1])
        distributions(x[:, 2], y[:, 2])
