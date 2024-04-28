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

from models import Encoder, Decoder, UNet
from helper import show_images, getPositionEncoding, forward_sample, Diffusion
from helper import distributions, norm, PSNR, ssim_loss


def fetch_data(image_size=128):
    dataset = torchvision.datasets.CelebA(
        root='/home/agam/Documents/git projects/pytorch_datasets', split='all',
        transform=transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ]), download=False)

    return dataset


class Loader:
    def __init__(self, batch_size=64, img_size=128, num_workers=8):
        dataset = fetch_data(img_size)
        self.data_loader = DataLoader(
            dataset=dataset, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, drop_last=True)
        self.iters = int(len(dataset) / batch_size)


def train_auto_enc(path, epochs=500, lr=1E-4, batch_size=64,
                   err_func=[nn.MSELoss(), nn.L1Loss(), PSNR(),
                             ssim_loss(channel=3, spatial_dims=2)],
                   lambdas=[1.0, 1.0, 0.5, 0.2], device='cpu', img_size=128):
    data = Loader(batch_size=batch_size, img_size=img_size)
    enc = Encoder(CH=3, latent=32).to(device)
    dec = Decoder(CH=3, latent=32).to(device)
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
            # x = torch.stack([dynamic_transforms(xi) for xi in x[0]], dim=0)
            x = dynamic_transforms(x[0])
            x = (norm(x).to(device) - 0.5) * 10

            z = norm(enc(x) - 0.5) * 10
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

        if (eps + 1) % 5 == 0:
            torch.save(enc.state_dict(), os.path.join(path, "CELEB-enc.pt"))
            torch.save(dec.state_dict(), os.path.join(path, "CELEB-dec.pt"))

    torch.save(enc.state_dict(), os.path.join(path, "CELEB-enc.pt"))
    torch.save(dec.state_dict(), os.path.join(path, "CELEB-dec.pt"))


def train(path, epochs=2000, lr=1E-6, batch_size=64, steps=400, emb=64,
          err_func=nn.L1Loss(), device='cpu', img_size=128):
    data = Loader(batch_size=batch_size, img_size=img_size)
    enc = Encoder(CH=3, latent=32).to(device)
    model = UNet(CH=32, emb=emb).to(device)
    # Crit = models.Critic(CH=128).cuda()
    enc.load_state_dict(torch.load(os.path.join(path, "CELEB-enc.pt")))
    try:
        model.load_state_dict(torch.load(os.path.join(path,
                                                      "CELEB-Autosave.pt")))
    except Exception:
        print('paramerts failed to load from last diffusion run')
    # try:
    #     model.load_state_dict(torch.load(os.path.join(path,
    #                                                   "CELEB-crit.pt")))
    # except Exception:
    #     print('paramerts failed to load from last critic run')
    optimizer = torch.optim.AdamW(model.parameters(), lr)
    # optimizerC = torch.optim.AdamW(Crit.parameters(),lr,betas=(0.5, 0.999))
    train_error = []
    avg_fact = data.iters
    tpop = range(1, steps)
    embds = getPositionEncoding(steps).to(device)

    for eps in range(epochs):
        print(f'Epoch {eps + 1}|{epochs}')
        model.train()
        enc.eval()
        running_error = 0.

        for itr, x in enumerate(tqdm(data.data_loader)):
            # optimizerC.zero_grad()
            # x = (norm(x[0]).to(device) - 0.5) * 10
            # z0 = (norm(enc(x)) - 0.5) * 10
            # t = torch.tensor(random.sample(tpop, len(z0)), device=device)
            # zt, noise = forward_sample(z0, t, steps, 'cos')
            # logits_real = critic(torch.cat((noise, zt), dim=1))
            # errC_real = logits_real.mean()
            # x = (norm(x[0]).to(device) - 0.5) * 10
            # z0 = (norm(enc(x)) - 0.5) * 10
            # t = torch.tensor(random.sample(tpop, len(z0)), device=device)
            # zt, _ = forward_sample(z0, t, steps, 'cos')
            # pred_noise = model(zt, embdt).detach()
            # logits_fake = critic(torch.cat((pred_noise, zt), dim=1))
            # errC_fake = logits_fake.mean()
            # penalty = grad_penalty(Crit, real, fake, Lambda_penalty)
            # error_critic = errC_fake - errC_real + penalty
            # errC.backward(retain_graph = True)
            # optimizerC.step()

            optimizer.zero_grad()
            x = (norm(x[0]).to(device) - 0.5) * 10
            z0 = (norm(enc(x)) - 0.5) * 10
            t = torch.tensor(random.sample(tpop, len(z0)), device=device)
            zt, noise = forward_sample(z0, t, steps, 'lin')
            embdt = embds[t]

            pred_noise = model(zt, embdt)
            # logits_fake = critic(torch.cat((pred_noise, zt), dim=1))
            error = err_func(noise, pred_noise)  # - logits_fake.mean()
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
                       os.path.join(path, "CELEB-Autosave.pt"))

        torch.save(model.state_dict(),
                   os.path.join(path, "CELEB-Autosave.pt"))


def fin(path, iterations=100, steps=1000):
    with torch.no_grad():
        diffusion = Diffusion(steps=steps, scheduler='lin')

        model = UNet(CH=32, emb=64).cuda()
        model.load_state_dict(
            torch.load(os.path.join(path, "CELEB-Autosave.pt")))

        dec = Decoder(CH=3, latent=32).cuda()
        dec.load_state_dict(
            torch.load(os.path.join(path, "CELEB-dec.pt")))

        dec.eval()
        model.eval()

        idx = torch.linspace(0, steps-1, 1, dtype=torch.int).cuda()
        print(idx)

        z = torch.randn((iterations, 32, 15, 15)).cuda()
        z_min, z_max = z.min().item(), z.max().item()
        print(z_min, z_max)

        imgs = []

        for t in trange(0, steps):
            z = torch.clamp(diffusion.backward(
                z, torch.tensor(steps - 1 - t), model), -5, 5)
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
    img_size = 128
    a = input('Train model from last checkpoint?(y/n)')
    if a == 'y':
        train_auto_enc(path=path, device='cuda',
                       batch_size=64, epochs=1000, lr=1E-4, img_size=img_size)
# =============================================================================
#         train(path=path, epochs=2000, img_size=img_size,
#               err_func=nn.L1Loss(), lr=1E-4, batch_size=64,
#               steps=1000, emb=64, device='cuda')
# =============================================================================
    else:
        y, zy = fin(path, iterations=itr, steps=1000, )
        if True:
            data = Loader(batch_size=itr, img_size=img_size)
            enc = Encoder(CH=3, latent=32).cpu()
            enc.load_state_dict(torch.load(os.path.join(path, "CELEB-enc.pt"),
                                           map_location='cpu'))
            for x in data.data_loader:
                x = (norm(x[0]) - 0.5) * 10
                z = (norm(enc(x)) - 0.5) * 10
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
