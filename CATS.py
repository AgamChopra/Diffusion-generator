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


def get_cats_avg():
    img = cv2.imread('R:/E/target.jpg',)
    x = cv2.resize(img, dsize=(512, 512), interpolation=cv2.INTER_CUBIC).T
    img = np.asarray([x[2], x[1], x[0]]).astype(dtype=np.float32)
    # img = 0.2126 * img[0] + 0.7152 * img[1] + 0.0722 * img[2]
    img = (img - img.min()) / (img.max() - img.min())
    img = (img - 0.5) * 6
    return img.reshape((1, 3, 512, 512))


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
    data = (data - 0.5) * 6
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

        if eps % 1 == 0:
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


STEPS = 1000


def fin(iterations=100, alpha=0.5):
    diffusion = Diffusion(steps=STEPS)
    x1 = torch.from_numpy(get_cats_avg()).to(dtype=torch.float, device='cuda')
    path = 'R:/E/parameters/'
    model = UNet(CH=3, emb=64, n=0.5).cuda()
    try:
        model.load_state_dict(torch.load(os.path.join(path,
                                                      "CATS-Autosave.pt")))
    except Exception:
        print('paramerts failed to load from last run')
    idx = torch.linspace(int(STEPS*alpha), 999, 16, dtype=torch.int).cuda()

    imgs = []
    x = torch.cat([torch.clamp(forward_sample(x1, torch.tensor(
        [STEPS - int(STEPS*alpha)], device='cuda'), STEPS, 0.0001,
        0.02)[0], -3, 3) for _ in range(iterations)], dim=0)

    for t in trange(int(STEPS*alpha), STEPS):
        x = torch.clamp(diffusion.backward(
            x, torch.tensor(STEPS - 1 - t), model), -3, 3)
        if t in idx:
            imgs.append(x[0:1])

    imgs = torch.cat(imgs, dim=0)
    show_images(imgs.cpu(), 16, 4, mode=False)
    show_images(x.cpu(), iterations, int(iterations ** 0.5), mode=False)
    return x.cpu()


if __name__ == '__main__':
    itr = 4
    a = input('Train model from last checkpoint?(y/n)')
    if a == 'y':
        train(path='R:/E/parameters/', epochs=1000,
              err_func=nn.HuberLoss(delta=1E-6), lr=1E-3, batch_size=2,
              steps=STEPS, n=0.5, emb=64, device='cuda')
    else:
        alpha = float(input('How dreamy? (0 to 1)'))
        y = fin(iterations=itr, alpha=alpha)
# =============================================================================
#         data = Loader(batch_size=itr)
#         for x in data.data_loader:
#             break
#         show_images(x.cpu(), itr, int(itr ** 0.5), mode=False)
#         distributions(x[:, 0], y[:, 0])
#         distributions(x[:, 1], y[:, 1])
#         distributions(x[:, 2], y[:, 2])
# =============================================================================
