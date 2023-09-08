"""
Created on Wed Aug 23 19:21:10 2023

@author: Agam Chopra
"""

import os
import random
import torch
import torchvision
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm, trange
from torchvision import transforms
from torch.utils.data import DataLoader

from models import UNet
from helper import show_images, getPositionEncoding, forward_sample, Diffusion
from helper import distributions


def fetch_data(image_size=64):
    dataset = torchvision.datasets.CelebA(
        root='E:/pytorch_datasets/data', split='all',
        transform=transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor()
        ]), download=True)

    return dataset


class Loader:
    def __init__(self, batch_size=64, img_size=64, num_workers=8):
        dataset = fetch_data(img_size)
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
                                                      "CELEB-Autosave.pt")))
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
            x0 = x[0].to(device)
            x0 = (((x0 - x0.min()) / (x0.max() - x0.min())) - 0.5) * 6
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
            plt.ylabel("Average L1 Error")
            plt.legend()
            plt.show()

            torch.save(model.state_dict(),
                       os.path.join(path, "CELEB-Autosave.pt"))


def fin(iterations=100):
    diffusion = Diffusion(steps=1000)

    path = 'R:/git projects/parameters/'
    model = UNet(CH=3, emb=64, n=1).cuda()
    model.load_state_dict(torch.load(os.path.join(path,
                                                  "CELEB-Autosave.pt")))
    idx = torch.linspace(0, 999, 16, dtype=torch.int).cuda()
    print(idx)

    x = torch.randn((iterations, 3, 64, 64)).cuda()
    imgs = []

    for t in trange(0, 1000):
        x = torch.clamp(diffusion.backward(
            x, torch.tensor(999 - t), model), -3, 3)
        if t in idx:
            imgs.append(x[0:1])

    imgs = torch.cat(imgs, dim=0)
    print(imgs.shape)
    show_images(imgs.cpu(), 16, 4)
    show_images(x.cpu(), iterations, int(iterations ** 0.5))
    return x.cpu()


if __name__ == '__main__':
    itr = 64
    a = input('Train model from last checkpoint?(y/n)')
    if a == 'y':
        train(path='R:/git projects/parameters/', epochs=2000,
              err_func=nn.HuberLoss(delta=0.06), lr=1E-4, batch_size=32,
              steps=1000, n=1, emb=64, device='cuda')
    else:
        y = fin(iterations=itr)
        data = Loader(batch_size=itr)
        for x in data.data_loader:
            x0 = x[0]
            x = (((x0 - x0.min()) / (x0.max() - x0.min())) - 0.5) * 6
            break
        distributions(x, y)
