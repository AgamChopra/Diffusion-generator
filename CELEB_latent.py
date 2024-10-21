"""
Created on Wed Aug 23 19:21:10 2023

@author: Agam Chopra
"""

import os
import random
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm, trange
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from PIL import Image

from model_v2 import Encoder, Decoder, UNet
from helper import show_images, getPositionEncoding, forward_sample, Diffusion
from helper import distributions, norm


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
            '/home/agam/Documents/git_projects/pytorch_datasets/celeba', transform=transform)
        self.data_loader = DataLoader(
            dataset=dataset, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, drop_last=True)
        self.iters = int(len(dataset) / batch_size)
        print('done.')


def train_auto_enc(path, epochs=500, lr=1E-3, batch_size=128,
                   err_func=[nn.MSELoss(), nn.L1Loss()],
                   lambdas=[0.65, 0.35], device='cpu', img_size=64):
    data = Loader(batch_size=batch_size, img_size=img_size)
    enc = Encoder(CH=3, latent=512).to(device)
    dec = Decoder(CH=3, latent=512).to(device)
    try:
        enc.load_state_dict(torch.load(os.path.join(path, "CELEB-enc.pt")))
        dec.load_state_dict(torch.load(os.path.join(path, "CELEB-dec.pt")))
    except Exception:
        print('paramerts failed to load from last run')
    optimizer = torch.optim.AdamW(
        list(enc.parameters()) + list(dec.parameters()), lr)
    train_error = []
    avg_fact = data.iters

    dynamic_transforms = transforms.Compose([
        transforms.RandomAutocontrast(),
        transforms.RandomAdjustSharpness(sharpness_factor=2),
        transforms.RandomHorizontalFlip(),
        transforms.RandomPerspective(
            interpolation=transforms.functional.InterpolationMode.BILINEAR),
        transforms.RandomAffine(
            degrees=25,
            interpolation=transforms.functional.InterpolationMode.BILINEAR)
    ])

    for eps in range(epochs):
        print(f'Epoch {eps + 1}|{epochs}')
        enc.train()
        dec.train()
        running_error = 0.

        for itr, x in enumerate(tqdm(data.data_loader)):
            optimizer.zero_grad()
            x = dynamic_transforms(x[0])
            x = norm(x).to(device)

            z = enc(x)
            x_ = dec(z)

            error = sum([lambdas[i] * err_func[i](x, x_)
                         for i in range(len(lambdas))])
            error.backward()
            optimizer.step()

            running_error += error.item()

            if itr >= avg_fact:
                break

        train_error.append(running_error / avg_fact)
        print(f'Average Error: {train_error[-1]}')

        if eps % 1 == 0:
            plt.figure(figsize=(10, 5))
            plt.title("AutoEncoder Training Error")
            plt.plot(train_error, label='Average Error per Epoch')
            plt.legend()
            plt.xlabel("Epochs")
            plt.ylabel("Average Error")
            plt.legend()
            plt.show()

            show_images(torch.cat((norm(x[:4]).cpu().detach(),
                                   norm(x_[0:4]).cpu().detach()), dim=0),
                        num_samples=8, cols=4, dpi=200)

        if (eps + 1) % 10 == 0:
            torch.save(enc.state_dict(), os.path.join(path, "CELEB-enc.pt"))
            torch.save(dec.state_dict(), os.path.join(path, "CELEB-dec.pt"))

    torch.save(enc.state_dict(), os.path.join(path, "CELEB-enc.pt"))
    torch.save(dec.state_dict(), os.path.join(path, "CELEB-dec.pt"))


def train(path, epochs=2000, lr=1E-4, batch_size=128,
          steps=1000, emb=64, device='cpu', img_size=64):

    data = Loader(batch_size=batch_size, img_size=img_size)

    enc = Encoder(CH=3, latent=512).to(device)
    enc.load_state_dict(torch.load(os.path.join(path, "CELEB-enc.pt")))

    model = UNet(CH=512, emb=emb, n=16).to(device)
    try:
        model.load_state_dict(torch.load(os.path.join(path,
                                                      "CELEB-Autosave-Diffusion.pt")))
    except Exception:
        print('paramerts failed to load from last diffusion run')

    optimizer = torch.optim.AdamW(model.parameters(), lr)
    train_error = []
    avg_fact = data.iters
    tpop = range(1, steps)
    embds = getPositionEncoding(steps).to(device)

    L1 = nn.L1Loss()
    L2 = nn.MSELoss()
    KL = nn.KLDivLoss(log_target=True)

    for eps in range(epochs):
        print(f'Epoch {eps + 1}|{epochs}')
        model.train()
        enc.eval()
        running_error = 0.

        for itr, x in enumerate(tqdm(data.data_loader)):
            optimizer.zero_grad()
            x = norm(x[0]).to(device)
            z0 = enc(x).detach()
            t = torch.tensor(random.sample(tpop, len(z0)), device=device)
            zt, noise = forward_sample(z0, t, steps, 'lin')
            embdt = embds[t]

            pred_noise = model(zt, embdt)
            error = 0.5 * L1(
                noise, pred_noise) + 0.3 * L2(
                    noise, pred_noise) + 0.2 * KL(
                        (norm(pred_noise) + 1e-6).log(),
                        (norm(noise) + 1e-6).log())
            error.backward()
            optimizer.step()

            running_error += error.item()

            if itr >= avg_fact:
                break

        train_error.append(running_error / avg_fact)
        print(f'Average Error: {train_error[-1]}')

        if eps % 2 == 0:
            plt.figure(figsize=(10, 5))
            plt.title("Training Noise Prediction Error")
            plt.plot(train_error, label='Average Error per Epoch')
            plt.legend()
            plt.xlabel("Epochs")
            plt.ylabel("Average Error")
            plt.legend()
            plt.show()

        if eps % 10 == 0:
            torch.save(model.state_dict(),
                       os.path.join(path, "CELEB-Autosave-Diffusion.pt"))

        torch.save(model.state_dict(),
                   os.path.join(path, "CELEB-Autosave-Diffusion.pt"))


@torch.no_grad()
def fin(path, iterations=100, steps=1000):
    with torch.no_grad():
        diffusion = Diffusion(steps=steps, scheduler='lin')

        model = UNet(CH=512, emb=64, n=16).cuda()
        try:
            model.load_state_dict(torch.load(os.path.join(path,
                                                          "CELEB-Autosave-Diffusion.pt")))
        except Exception:
            print('paramerts failed to load from last diffusion run')

        dec = Decoder(CH=3, latent=512).cuda()
        dec.load_state_dict(
            torch.load(os.path.join(path, "CELEB-dec.pt")))

        dec.eval()
        model.eval()

        idx = torch.linspace(0, steps-1, 1, dtype=torch.int).cuda()
        print(idx)

        z = (norm(torch.randn((iterations, 512, 15, 15)).cuda()) - 0.5) * 2
        z_min, z_max = z.min().item(), z.max().item()
        print(z_min, z_max)

        imgs = []

        for t in trange(0, steps):
            z = torch.clamp(diffusion.backward(
                z, torch.tensor(steps - 1 - t), model), -1, 1)
            if t in idx:
                imgs.append(dec(z[0:1]))

        print(z.min().item(), z.max().item(), z.mean().item())
        x = dec(z)
        print(x.min().item(), x.max().item(), x.mean().item())
        imgs = torch.cat(imgs, dim=0)
        print(imgs.shape)
        show_images(imgs.cpu(), 1, 4)
        show_images(norm(x).cpu(), iterations, int(iterations ** 0.5))
        return x.cpu(), z.cpu()


if __name__ == '__main__':
    path = os.path.abspath(__file__)[:-15]
    os.chdir(path)
    print(path)

    itr = 16
    img_size = 64
    a = input('Train model from last checkpoint?(y/n)')
    if a == 'y':
        a = input('Train autoencoder(a) or diffusion(d)?(a/d)')
        if a == 'a':
            train_auto_enc(path=path, device='cuda',
                           batch_size=64, epochs=1000, lr=1E-5,
                           img_size=img_size)
        else:
            train(path=path, epochs=2000, img_size=img_size, lr=1E-4, batch_size=64,
                  steps=1000, emb=64, device='cuda')

    else:
        y, zy = fin(path, iterations=itr, steps=1000, )
        if True:
            data = Loader(batch_size=itr, img_size=img_size)
            enc = Encoder(CH=3, latent=512).cpu()
            enc.load_state_dict(torch.load(os.path.join(path, "CELEB-enc.pt"),
                                           map_location='cpu'))
            for x in data.data_loader:
                x = norm(x[0])
                z = enc(x)
                break
            distributions(x.detach(), y.detach(), th=5, bins=128)
            distributions(z.detach(), zy.detach(), th=5, bins=256)
            print(z.detach().shape, zy.detach().shape)
            show_images(torch.unsqueeze(torch.mean(
                z.detach(), dim=1), dim=1), itr, int(itr ** 0.5))
            show_images(torch.unsqueeze(torch.mean(
                zy.detach(), dim=1), dim=1), itr, int(itr ** 0.5))
            show_images(torch.unsqueeze(torch.max(
                z.detach(), dim=1)[0], dim=1), itr, int(itr ** 0.5))
            show_images(torch.unsqueeze(torch.max(
                zy.detach(), dim=1)[0], dim=1), itr, int(itr ** 0.5))
            show_images(torch.unsqueeze(
                z[:, 0].detach(), dim=1), itr, int(itr ** 0.5))
            show_images(torch.unsqueeze(
                zy[:, 0].detach(), dim=1), itr, int(itr ** 0.5))
