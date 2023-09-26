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
from torchvision import transforms

from models import UNet
from helper import show_images, getPositionEncoding, forward_sample, Diffusion
from helper import distributions


EPS = 1000
LR = 1E-3
ITR = 9
N = 0.25
SCH = 'lin'
STEPS = 400 if SCH == 'cos' else 1000


TRF = transforms.Compose(
    [transforms.CenterCrop(162*2), transforms.Resize(128)])


def Trf(x):
    x = torch.cat([transforms.functional.rotate(
        x[i:i+1], random.randint(-45, 45),
        interpolation=transforms.InterpolationMode.BILINEAR) for i in range(
            len(x))], dim=0)
    return TRF(x)


def load_cats():
    cat_list = []
    for i in trange(5653):
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
    data = (data - 0.5) * 4.7 * 2
    return data


class Loader:
    def __init__(self, batch_size=32, num_workers=4):
        dataset = fetch_data()
        self.data_loader = DataLoader(
            dataset=dataset, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, drop_last=True)
        self.iters = int(len(dataset) / batch_size)


def train(path, epochs=2000, lr=1E-6, batch_size=64, steps=1000, n=1, emb=64,
          err_func=nn.L1Loss(), device='cpu', scheduler='lin'):
    data = Loader(batch_size=batch_size)
    model = UNet(CH=3, emb=emb, n=n).to(device)
    savefile = "CATS-Autosave-lin.pt" if scheduler == 'lin' else \
        "CATS-Autosave-cos.pt"
    try:
        model.load_state_dict(torch.load(
            os.path.join(path, savefile)))
    except Exception:
        print('paramerts failed to load from last run')
    optimizer = torch.optim.AdamW(model.parameters(), lr)

    train_error = []
    avg_fact = data.iters
    tpop = range(1, steps)
    embds = getPositionEncoding(steps).to(device)

    for eps in range(epochs):
        if (eps + 1) % 250 == 0:
            for param in optimizer.param_groups:
                param['lr'] *= 0.1
            print('***updated learning rate***')

        print(f'Epoch {eps + 1}|{epochs}')

        model.train()
        running_error = 0.

        for itr, x in enumerate(tqdm(data.data_loader)):
            optimizer.zero_grad()
            x0 = Trf(x).to(device)
            t = torch.tensor(random.sample(tpop, len(x0)), device=device)
            xt, noise = forward_sample(x0, t, steps, scheduler=scheduler)
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

        if eps % 250 == 0:
            plt.figure(figsize=(10, 5))
            plt.title("Training Noise Prediction Error")
            plt.plot(train_error, label='Average Error per Epoch')
            plt.legend()
            plt.xlabel("Epochs")
            plt.ylabel("Average Error")
            plt.legend()
            plt.show()

            torch.save(model.state_dict(),
                       os.path.join(path, savefile))

    torch.save(model.state_dict(),
               os.path.join(path, savefile))


@torch.no_grad()
def fin(iterations=100, scheduler='lin'):
    diffusion = Diffusion(steps=STEPS, scheduler=scheduler)
    path = 'R:/E/parameters/'
    savefile = "CATS-Autosave-lin.pt" if scheduler == 'lin' else \
        "CATS-Autosave-cos.pt"

    model = UNet(CH=3, emb=64, n=N).cuda()

    try:
        model.load_state_dict(torch.load(
            os.path.join(path, savefile)))
    except Exception:
        print('paramerts failed to load from last run')

    model.eval()

    idx = torch.linspace(0, STEPS - 1, 64, dtype=torch.int).cuda()
    imgs = []

    x = torch.randn((itr, 3, 128, 128), device='cuda')
    x_min, x_max = x.min().item(), x.max().item()
    print(x_min, x_max)

    imgs.append(x[0:1])
    # show_images(x.cpu(), 4, 2, mode=False, size=(3, 4.5), dpi=256)

    for t in trange(0, STEPS):
        x = torch.clamp(diffusion.backward(
            x, torch.tensor(STEPS - t - 1), model), -4.7, 4.7)
        # show_images(x.cpu(), 4, 2, mode=False, size=(3, 4.5), dpi=256)
        if t in idx[1:]:
            imgs.append(x[0:1])

    imgs = torch.cat(imgs, dim=0)
    show_images(imgs.cpu(), 64, 8, mode=False)
    show_images(x.cpu(), iterations, int(iterations ** 0.5), mode=False)
    return x.cpu()


if __name__ == '__main__':
    itr = ITR
    scheduler = SCH

    a = input('Train model from last checkpoint?(y/n)')
    if a == 'y':
        train(path='R:/E/parameters/', epochs=EPS,
              err_func=nn.L1Loss(), lr=LR, batch_size=itr,
              steps=STEPS, n=N, emb=64, device='cuda', scheduler=scheduler)
    else:
        y = fin(iterations=itr, scheduler=scheduler)
        print(y.shape, y.max(), y.min())
        data = Loader(batch_size=itr)
        for x in data.data_loader:
            break
        x = Trf(x)
        print(x.shape, x.max(), x.min())
        show_images(x.cpu(), itr, int(itr ** 0.5), mode=False)
        distributions(x[:, 0], y[:, 0])
        distributions(x[:, 1], y[:, 1])
        distributions(x[:, 2], y[:, 2])

