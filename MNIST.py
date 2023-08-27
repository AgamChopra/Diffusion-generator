"""
Created on Wed Aug 23 19:21:10 2023

@author: Agam Chopra
"""

import torch
import torchvision
import torch.nn as nn
import matplotlib.pyplot as plt
import random
from math import ceil
from tqdm import trange
import os

from models import UNet
from helper import show_images, getPositionEncoding, forward_sample, Diffusion


def fetch_data():
    data = torchvision.datasets.MNIST(
        root='./data', train=True, download=True, transform=None).data
    data = ((data[:, None] / 255) * 2) - 1
    return data


class Dataloader:
    def __init__(self, batch_size=64):
        self.data = fetch_data()
        self.max_id = len(self.data) - 1
        self.id = 0
        self.batch = batch_size
        self.range = range(0, len(self.data))
        self.randomize()

    def randomize(self):
        idx = random.sample(self.range, self.max_id + 1)
        self.data = self.data[idx]

    def sample(self):
        if self.id + self.batch > self.max_id:
            if self.id < self.max_id:
                batch_data = self.data[self.id:]
            else:
                batch_data = self.data[self.id:self.id + 1]
            self.id = 0
            self.randomize()
        else:
            batch_data = self.data[self.id:self.id+self.batch]
        return batch_data


def train(path, epochs=2000, lr=1E-6, batch_size=64, steps=1000, n=1, emb=64,
          err_func=nn.L1Loss(), device='cpu'):
    data = Dataloader(batch_size=batch_size)
    model = UNet(CH=data.data.shape[1], emb=emb, n=n).to(device)
    try:
        model.load_state_dict(torch.load(os.path.join(path,
                                                      "diffusion-MNIST-Autosave.pt")))
    except Exception:
        print('paramerts failed to load from last run')
    optimizer = torch.optim.Adam(model.parameters(), lr)
    train_error = []
    avg_fact = ceil(len(data.data) / batch_size)
    tpop = range(1, steps)
    embds = getPositionEncoding(steps).to(device)

    for eps in range(epochs):
        print(f'Epoch {eps + 1}|{epochs}')
        model.train()
        running_error = 0.

        for _ in trange(0, avg_fact):
            optimizer.zero_grad()
            x0 = data.sample().to(device)
            t = torch.tensor(random.sample(tpop, len(x0)), device=device)
            xt, noise = forward_sample(x0, t, steps, 0.0001, 0.02)
            embdt = embds[t]

            pred_noise = model(xt, embdt)
            error = err_func(noise, pred_noise)
            error.backward()
            optimizer.step()

            running_error += error.item()

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
                       os.path.join(path, "diffusion-MNIST-Autosave.pt"))


def fin(iterations=100):
    diffusion = Diffusion(steps=1000)

    path = 'T:/github/Diffusion-generator/parameters/'
    model = UNet(emb=64, n=8).cuda()
    model.load_state_dict(torch.load(os.path.join(path,
                                                  "diffusion-MNIST-Autosave.pt")))

    for _ in range(100):
        x = torch.randn((1, 1, 28, 28)).cuda()
        idx = torch.linspace(0, 999, 16, dtype=torch.int).cuda()
        imgs = []

        for t in range(0, 1000):
            x = torch.clamp(diffusion.backward(
                x, torch.tensor(999 - t), model), -1, 1)
            if t in idx:
                imgs.append(x)
        imgs = torch.cat(imgs, dim=0)
        show_images(imgs.cpu(), 16, 4)


if __name__ == '__main__':
    a = input('Train model from last checkpoint?(y/n)')
    if a == 'y':
        train(path='T:/github/Diffusion-generator/parameters/', epochs=10000,
              lr=1E-6, batch_size=256, steps=1000, n=8, emb=64, device='cuda')
    else:
        fin()
