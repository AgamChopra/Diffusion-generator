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
from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import DataLoader

from models import UNet
from helper import show_images, getPositionEncoding, forward_sample, Diffusion


def fetch_data(image_size=64):
    dataset = torchvision.datasets.CelebA(
        root='./data', split='all', transform=transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor()
        ]), download=True)

    return dataset


class Loader:
    def __init__(self, batch_size=64, img_size=64, num_workers=4):
        dataset = fetch_data(img_size)
        self.data_loader = DataLoader(
            dataset=dataset, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, drop_last=True)
        self.iters = int(len(dataset) / batch_size)


def train(path, epochs=2000, lr=1E-6, batch_size=64, steps=1000, n=1, emb=64,
          err_func=nn.L1Loss(), device='cpu'):
    data = Loader()
    model = UNet(CH=3, emb=emb, n=n).to(device)
    try:
        model.load_state_dict(torch.load(os.path.join(path,
                                                      "CELEB-Autosave.pt")))
    except Exception:
        print('paramerts failed to load from last run')
    optimizer = torch.optim.Adam(model.parameters(), lr)
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

    path = 'T:/github/Diffusion-generator/parameters/'
    model = UNet(CH=3, emb=64, n=8).cuda()
    model.load_state_dict(torch.load(os.path.join(path,
                                                  "CELEB-Autosave.pt")))

    for _ in range(100):
        x = torch.randn((1, 3, 64, 64)).cuda()
        idx = torch.linspace(0, 999, 16, dtype=torch.int).cuda()
        imgs = []

        for t in range(0, 1000):
            x = torch.clamp(diffusion.backward(
                x, torch.tensor(999 - t), model), 0, 1)
            if t in idx:
                imgs.append(x)
        imgs = torch.cat(imgs, dim=0)
        show_images(imgs.cpu(), 16, 4)


if __name__ == '__main__':
    a = input('Train model from last checkpoint?(y/n)')
    if a == 'y':
        train(path='T:/github/Diffusion-generator/parameters/', epochs=100,
              lr=1E-3, batch_size=64, steps=1000, n=8, emb=64, device='cuda')
    else:
        fin()