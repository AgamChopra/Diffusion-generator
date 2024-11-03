"""
Created on Wed Aug 23 19:21:10 2023

@author: Agam Chopra
"""

import os
import random
import torch
import torch.nn as nn
import torch.amp as amp
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import torch.optim.lr_scheduler as lr_scheduler
import matplotlib.pyplot as plt
from tqdm import tqdm, trange
import pandas as pd
from PIL import Image

from model_v2 import Encoder, Decoder, UNet, reparameterize
from helper import show_images, getPositionEncoding, forward_sample, Diffusion
from helper import distributions, norm  # , plot_grad_hist
from helper import KL_Loss, ssim_loss


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
            '/home/agam/Documents/git_projects/pytorch_datasets/celeba',
            transform=transform)
        self.data_loader = DataLoader(
            dataset=dataset, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, drop_last=True)
        self.iters = int(len(dataset) / batch_size)
        print('done.')


def train_auto_enc(path, epochs=500, lr=1E-3, batch_size=128,
                   device='cpu', img_size=64):
    data = Loader(batch_size=batch_size, img_size=img_size)
    enc = Encoder(CH=3, latent=128, num_groups=8).to(device)
    dec = Decoder(CH=3, latent=128, num_groups=8).to(device)

    try:
        enc.load_state_dict(torch.load(os.path.join(path, "CELEB-enc.pt")))
        dec.load_state_dict(torch.load(os.path.join(path, "CELEB-dec.pt")))
    except Exception:
        print('paramerts failed to load from last run')

    optimizer = torch.optim.AdamW(
        list(enc.parameters()) + list(dec.parameters()), lr)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = amp.GradScaler('cuda')

    train_error = []
    MSE = nn.MSELoss()
    MAE = nn.L1Loss()
    SSIM = ssim_loss()
    KL = KL_Loss()

    dynamic_transforms = transforms.Compose([
        # transforms.RandomAdjustSharpness(sharpness_factor=1.2),
        transforms.RandomHorizontalFlip(),
        # transforms.RandomVerticalFlip(),
        # transforms.RandomPerspective(
        #     interpolation=transforms.functional.InterpolationMode.BILINEAR),
        # transforms.RandomAffine(degrees=45,
        #                         interpolation=transforms.functional.InterpolationMode.BILINEAR),
        # transforms.RandomResizedCrop(
        #     img_size, scale=(0.8, 1.2),
        #     interpolation=transforms.functional.InterpolationMode.BILINEAR),
        # transforms.ColorJitter(brightness=0.1, contrast=0.1,
        #                        saturation=0.1)  # Color jitter
    ])

    for eps in range(epochs):
        enc.train()
        dec.train()
        running_error = 0.

        for itr, x in enumerate(tqdm(data.data_loader)):
            optimizer.zero_grad()
            if random.random() > 0.1:
                x = dynamic_transforms(x[0])
            else:
                x = x[0]
            x = norm(x).to(device)

            with amp.autocast('cuda'):
                mu, logvar = enc(x)
                x_ = dec(mu, logvar)

                # Adjust loss weighting
                error = 2 * SSIM(x, x_) + 10 * MSE(x, x_) + \
                    1e-6 * KL(mu, logvar) + 10 * MAE(x, x_)

            scaler.scale(error).backward()
            torch.nn.utils.clip_grad_norm_(
                list(enc.parameters()) + list(dec.parameters()), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            running_error += error.item()

        scheduler.step()
        train_error.append(running_error / data.iters)
        print(f'Epoch [{eps + 1}/{epochs}], Average Error: {train_error[-1]}')

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

        if (eps + 1) % 1 == 0:
            torch.save(enc.state_dict(), os.path.join(path, "CELEB-enc.pt"))
            torch.save(dec.state_dict(), os.path.join(path, "CELEB-dec.pt"))

    torch.save(enc.state_dict(), os.path.join(path, "CELEB-enc.pt"))
    torch.save(dec.state_dict(), os.path.join(path, "CELEB-dec.pt"))


def train(path, epochs=2000, lr=1E-4, batch_size=128,
          steps=1000, emb=64, device='cpu', img_size=64):

    data = Loader(batch_size=batch_size, img_size=img_size)

    with torch.no_grad():
        enc = Encoder(CH=3, latent=128, num_groups=8).eval().to(device)
        enc.load_state_dict(torch.load(os.path.join(path, "CELEB-enc.pt"),
                                       map_location=device))

    model = UNet(CH=128, emb=emb, n=8, num_groups=4).to(device)
    try:
        model.load_state_dict(torch.load(
            os.path.join(path, "CELEB-Autosave-Diffusion.pt")))
    except Exception:
        print('Parameters failed to load from the last diffusion run')

    optimizer = torch.optim.AdamW(model.parameters(), lr)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = amp.GradScaler('cuda')

    train_error = []
    train_error_h = []
    train_error_kl = []
    avg_fact = data.iters
    tpop = range(1, steps)
    embds = getPositionEncoding(steps).to(device)

    MSE = nn.MSELoss()
    MAE = nn.L1Loss()
    KL = nn.KLDivLoss(log_target=True, reduction='batchmean')

    dynamic_transforms = transforms.Compose([
        transforms.RandomHorizontalFlip(),
    ])

    for eps in range(epochs):
        print(f'Epoch {eps + 1}/{epochs}')
        model.train()
        enc.eval()
        running_error = 0.
        running_error_h = 0.
        running_error_kl = 0.

        lr = optimizer.param_groups[0]['lr']
        print(f'    Learning Rate: {lr}')

        for itr, x in enumerate(tqdm(data.data_loader)):
            optimizer.zero_grad()
            if random.random() > 0.1:
                x = dynamic_transforms(x[0])
            else:
                x = x[0]
            x = norm(x).to(device)
            mu, logvar = enc(x)
            z0 = reparameterize(mu, logvar).detach()
            t = torch.tensor(random.sample(tpop, len(z0)), device=device)
            zt, noise = forward_sample(z0, t, steps, 'lin')
            embdt = embds[t]

            with amp.autocast('cuda'):
                pred_noise = model(zt, embdt)

                kl_error = KL(nn.functional.log_softmax(pred_noise, dim=0),
                              nn.functional.log_softmax(noise, dim=0))

                pix_error = MSE(noise, pred_noise) + MAE(noise, pred_noise)

                error = 10 * pix_error + 1E-6 * kl_error

            # Backward pass with mixed precision
            scaler.scale(error).backward()
            torch.nn.utils.clip_grad_norm_(list(model.parameters()),
                                           max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            running_error += error.item()
            running_error_h += pix_error.item()
            running_error_kl += kl_error.item()

            if itr >= avg_fact:
                break

        scheduler.step()

        train_error.append(running_error / avg_fact)
        train_error_h.append(running_error_h / avg_fact)
        train_error_kl.append(running_error_kl / avg_fact)
        print(f'Average Total Error: {train_error[-1]}, Pixel Error: {
              train_error_h[-1]}, KL Divergance Error: {train_error_kl[-1]}')
        # plot_grad_hist(model.named_parameters())

        if eps % 1 == 0:
            plt.figure(figsize=(10, 5))
            plt.title("Training Noise Prediction Error")
            plt.plot(train_error, label='Average Error')
            # plt.plot(train_error_h, label='Average Huber')
            # plt.plot(train_error_kl, label='Average KL_div')
            plt.legend()
            plt.xlabel("Epochs")
            plt.ylabel("Average Error")
            plt.legend()
            plt.show()

        torch.save(model.state_dict(), os.path.join(
            path, "CELEB-Autosave-Diffusion.pt"))


@torch.no_grad()
def fin(path, iterations=100, steps=1000):
    with torch.no_grad():
        diffusion = Diffusion(steps=steps, scheduler='lin')

        model = UNet(CH=128, emb=64, n=8, num_groups=4).cuda()
        try:
            model.load_state_dict(
                torch.load(os.path.join(path, "CELEB-Autosave-Diffusion.pt")))
        except Exception:
            print('paramerts failed to load from last diffusion run')

        dec = Decoder(CH=3, latent=128, num_groups=8).cuda()
        dec.load_state_dict(
            torch.load(os.path.join(path, "CELEB-dec.pt")))

        dec.eval()
        model.eval()

        idx = torch.linspace(0, steps-1, 16, dtype=torch.int).cuda()
        print(idx)

        # bounded_gaussian_noise((iterations, 512, 8, 8)).cuda()
        z = torch.randn((iterations, 128, 8, 8)).cuda()
        z_min, z_max = z.min().item(), z.max().item()
        print(z_min, z_max)

        imgs = []

        for t in trange(0, steps):
            z = torch.clamp(diffusion.backward(
                z, torch.tensor(steps - 1 - t), model), -5.5, 5.5)
            if t in idx:
                imgs.append(dec(z[0:1]))

        print(z.min().item(), z.max().item(), z.mean().item())
        x = norm(dec(z))
        print(x.min().item(), x.max().item(), x.mean().item())
        imgs = torch.cat(imgs, dim=0)
        print(imgs.shape)
        show_images(norm(imgs).cpu(), 16, 4)
        show_images(norm(x).cpu(), iterations, int(iterations ** 0.5))
        return x.cpu(), z.cpu()


if __name__ == '__main__':
    path = os.path.abspath(__file__)[:-15]
    os.chdir(path)
    print(path)

    itr = 64
    img_size = 64
    steps = 1000

    a = input('Train model from last checkpoint?(y/n)')
    if a == 'y':
        a = input('Train autoencoder(a) or diffusion(d)?(a/d)')
        if a == 'a':
            train_auto_enc(path=path, device='cuda',
                           batch_size=128, epochs=1000, lr=1E-5,
                           img_size=img_size)
        else:
            train(path=path, epochs=1000, img_size=img_size, lr=1E-5,
                  batch_size=16, steps=steps, emb=64, device='cuda')

    else:
        with torch.no_grad():
            y, zy = fin(path, iterations=itr, steps=steps)

            if True:
                data = Loader(batch_size=itr, img_size=img_size)
                enc = Encoder(CH=3, latent=128, num_groups=8)
                enc.load_state_dict(torch.load(os.path.join(path, "CELEB-enc.pt"),
                                               map_location='cpu'))
                dec = Decoder(CH=3, latent=128, num_groups=8)
                dec.load_state_dict(torch.load(os.path.join(path, "CELEB-dec.pt"),
                                               map_location='cpu'))
                for x in data.data_loader:
                    x = norm(x[0])
                    mu, logvar = enc(x)
                    z = reparameterize(mu, logvar)
                    x_ = dec(mu, logvar)
                    break
                distributions(x_.detach(), y.detach(), th=5, bins=256)
                distributions(z.detach(), zy.detach(), th=5, bins=256)

                show_images(torch.cat((norm(x[:4]).cpu().detach(),
                                       norm(x_[0:4]).cpu().detach(),
                                       norm(y[0:4]).cpu().detach()), dim=0),
                            num_samples=12, cols=4, dpi=200)

                show_images(z.detach().cpu().mean(dim=0).unsqueeze(dim=1),
                            z.shape[1], int(z.shape[1] ** 0.5))
                show_images(zy.detach().cpu().mean(dim=0).unsqueeze(dim=1),
                            zy.shape[1], int(zy.shape[1] ** 0.5))
                show_images(torch.abs(z.mean(dim=0) - zy.mean(dim=0)).detach().cpu().unsqueeze(dim=1),
                            zy.shape[1], int(zy.shape[1] ** 0.5))
