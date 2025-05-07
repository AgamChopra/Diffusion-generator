import os
import random
import torch
import torch.nn as nn
import torch.amp as amp
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import torch.optim.lr_scheduler as lr_scheduler
from tqdm import tqdm, trange
import pandas as pd
from PIL import Image

from model_v2 import VQEncoder, VQDecoder, UNet, VectorQuantizerEMA
from helper import show_images, getPositionEncoding, forward_sample, Diffusion
from helper import distributions, norm, ssim_loss


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


def train_auto_enc(path, epochs=500, lr=1e-3, batch_size=128,
                   device='cpu', img_size=64):

    data = Loader(batch_size=batch_size, img_size=img_size)

    enc = VQEncoder(CH=3, latent=128, num_groups=8).to(device)
    dec = VQDecoder(CH=3, latent=128, num_groups=8).to(device)
    vq = VectorQuantizerEMA(num_embeddings=512, embedding_dim=128,
                            beta=0.25, decay=0.99).to(device)

    try:
        enc.load_state_dict(torch.load(os.path.join(path, 'CELEB-enc.pt')))
        dec.load_state_dict(torch.load(os.path.join(path, 'CELEB-dec.pt')))
        vq .load_state_dict(torch.load(os.path.join(path, 'CELEB-vq.pt')))
    except Exception:
        print('No previous VQ‑VAE checkpoint found – starting fresh.')

    opt = torch.optim.AdamW(list(enc.parameters()) + list(dec.parameters()),
                            lr=lr, weight_decay=1e-4)
    sched = lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    scaler = amp.GradScaler(device if device.startswith('cuda') else 'cpu')

    L1 = nn.L1Loss()
    SSIM = ssim_loss()
    λ_l1, λ_ssim = 10.0, 2.0

    for ep in range(epochs):
        enc.train()
        dec.train()
        running = 0.0

        for x, _ in tqdm(data.data_loader, leave=False):
            x = norm(x).to(device)

            with amp.autocast(device):
                z_e = enc(x)
                z_q, vq_l, _ = vq(z_e)
                x_hat = dec(z_q)

                loss = (
                    λ_l1 * L1(x_hat, x) +
                    λ_ssim * SSIM(x_hat, x) +
                    vq_l
                )

            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(list(enc.parameters()) +
                                           list(dec.parameters()), 1.0)
            scaler.step(opt)
            scaler.update()
            opt.zero_grad()
            running += loss.item()

        sched.step()
        print(f'epoch {ep+1}/{epochs} – loss {running / data.iters:.4f}')

        if (ep+1) % 1 == 0:
            show_images(torch.cat((norm(x[:4]).cpu(),
                                   norm(x_hat[:4]).cpu())), 8, 4, 200)
            torch.save(enc.state_dict(), os.path.join(path, 'CELEB-enc.pt'))
            torch.save(dec.state_dict(), os.path.join(path, 'CELEB-dec.pt'))
            torch.save(vq .state_dict(), os.path.join(path, 'CELEB-vq.pt'))


def train(path, epochs=2000, lr=1e-4, batch_size=128,
          steps=1000, emb=64, device='cpu', img_size=64):

    data = Loader(batch_size=batch_size, img_size=img_size)

    enc = VQEncoder(CH=3, latent=128, num_groups=8).eval().to(device)
    enc.load_state_dict(torch.load(os.path.join(path, 'CELEB-enc.pt'),
                                   map_location=device))

    unet = UNet(CH=128, emb=emb, n=8, num_groups=4).to(device)
    try:
        unet.load_state_dict(torch.load(os.path.join(
            path, 'CELEB-Autosave-Diffusion.pt')))
    except Exception:
        print('No previous UNet ckpt – fresh start.')

    opt = torch.optim.AdamW(unet.parameters(), lr=lr, weight_decay=1e-4)
    sched = lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    scaler = amp.GradScaler(device if device.startswith('cuda') else 'cpu')

    MSE, L1 = nn.MSELoss(), nn.L1Loss()
    embds = getPositionEncoding(steps).to(device)
    tpop = range(1, steps)

    for ep in range(epochs):
        unet.train()
        enc.eval()
        tot = 0.0

        for x, _ in tqdm(data.data_loader, leave=False):
            x = norm(x).to(device)
            z0 = enc(x).detach()
            t = torch.tensor(random.sample(tpop, len(z0)), device=device)
            zt, noise = forward_sample(z0, t, steps, 'lin')
            te = embds[t]

            with amp.autocast(device):
                pred = unet(zt, te)
                loss = MSE(pred, noise) + L1(pred, noise)

            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(unet.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            opt.zero_grad()
            tot += loss.item()

        sched.step()
        print(f'epoch {ep+1}/{epochs} – UNet loss {tot / data.iters:.4f}')
        torch.save(unet.state_dict(),
                   os.path.join(path, 'CELEB-Autosave-Diffusion.pt'))


@torch.no_grad()
def fin(path, iterations=100, steps=1000):
    diff = Diffusion(steps=steps, scheduler='lin')

    unet = UNet(CH=128, emb=64, n=8, num_groups=4).cuda()
    unet.load_state_dict(torch.load(os.path.join(
        path, 'CELEB-Autosave-Diffusion.pt')))
    unet.eval()

    dec = VQDecoder(CH=3, latent=128, num_groups=8).cuda()
    dec.load_state_dict(torch.load(os.path.join(path, 'CELEB-dec.pt')))
    dec.eval()

    vq = VectorQuantizerEMA(num_embeddings=512, embedding_dim=128).cuda()
    vq.load_state_dict(torch.load(os.path.join(path, 'CELEB-vq.pt')))
    vq.eval()

    z = torch.randn((iterations, 128, 8, 8), device='cuda')

    for t in trange(steps, desc='denoising'):
        z = diff.backward(z, torch.tensor(steps-1-t, device='cuda'), unet)
        z = z.clamp_(-5.5, 5.5)

    z_q, _, _ = vq(z)
    x = norm(dec(z_q))

    show_images(x.cpu(), iterations, int(iterations**0.5))
    return x.cpu(), z.cpu()


if __name__ == '__main__':
    path = os.path.abspath(__file__)[:-24]
    os.chdir(path)
    print(f'working dir → {path}')

    itr = 16
    img_size = 64
    steps = 1000

    cmd = input('Resume training from last checkpoint? (y/n) : ').lower()
    if cmd == 'y':
        which = input(
            'Train auto‑encoder (a) or diffusion model (d) ? (a/d) : ').lower()
        if which == 'a':
            train_auto_enc(path=path, device='cuda',
                           batch_size=128, epochs=1000, lr=1e-3,
                           img_size=img_size)
        else:
            train(path=path, epochs=1000, img_size=img_size, lr=1e-3,
                  batch_size=16, steps=steps, emb=64, device='cuda')

    else:
        y, zy = fin(path, iterations=itr, steps=steps)
        data = Loader(batch_size=itr, img_size=img_size)
        enc = VQEncoder(CH=3, latent=128, num_groups=8)
        enc.load_state_dict(torch.load(os.path.join(path, 'CELEB-enc.pt'),
                                       map_location='cpu'))
        dec = VQDecoder(CH=3, latent=128, num_groups=8)
        dec.load_state_dict(torch.load(os.path.join(path, 'CELEB-dec.pt'),
                                       map_location='cpu'))
        vq = VectorQuantizerEMA(num_embeddings=512, embedding_dim=128)
        vq.load_state_dict(torch.load(os.path.join(path, 'CELEB-vq.pt'),
                                      map_location='cpu'))
        enc.eval()
        dec.eval()
        vq.eval()

        for xb, _ in data.data_loader:
            x = norm(xb)
            z_e = enc(x)
            z_q, _, _ = vq(z_e)
            x_ = dec(z_q)
            break

        distributions(x_.detach(), y.detach(),  th=5, bins=256)
        distributions(z_e.detach(), zy.detach(), th=5, bins=256)

        show_images(torch.cat((norm(x[:4]).cpu(),
                               norm(x_[:4]).cpu(),
                               norm(y[:4]).cpu()), dim=0),
                    num_samples=12, cols=4, dpi=200)

        show_images(z_e.mean(0, keepdim=True).cpu(),   z_e.shape[1],
                    int(z_e.shape[1]**0.5))
        show_images(zy.mean(0, keepdim=True).cpu(),    zy.shape[1],
                    int(zy.shape[1]**0.5))
        show_images(torch.abs(z_e.mean(0) - zy.mean(0)).unsqueeze(0).cpu(),
                    zy.shape[1], int(zy.shape[1]**0.5))
