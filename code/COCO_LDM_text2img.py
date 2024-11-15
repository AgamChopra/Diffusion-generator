"""
Created on Wed Aug 23 19:21:10 2023

@author: Agam Chopra
"""
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.amp as amp
import torch.multiprocessing as mp
from torchvision import transforms
from torchvision.datasets import CocoCaptions
from torch.utils.data import DataLoader
import torch.optim.lr_scheduler as lr_scheduler
import matplotlib.pyplot as plt
from tqdm import tqdm, trange
from transformers import CLIPTokenizer, CLIPModel

from model_v2 import Encoder, Decoder, reparameterize
from model_v3 import UNet
from helper import show_images, getPositionEncoding, forward_sample, Diffusion
from helper import distributions, norm
from helper import KL_Loss, ssim_loss


print('cuda detected:', torch.cuda.is_available())
torch.set_printoptions(precision=6)

# Set start method to spawn for CUDA compatibility
mp.set_start_method('spawn', force=True)

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


class Loader:
    def __init__(self, batch_size=32, img_size=256, num_workers=8, device='cpu'):
        coco_root = '/home/agam/Documents/git_projects/pytorch_datasets/coco/train2017/'
        ann_file = '/home/agam/Documents/git_projects/pytorch_datasets/coco/annotations/captions_train2017.json'
        clip_model_name = "openai/clip-vit-base-patch32"

        self.device = device

        print('\rloading CLIP', end='')
        self.clip_model = CLIPModel.from_pretrained(clip_model_name).to(device)
        self.clip_tokenizer = CLIPTokenizer.from_pretrained(clip_model_name)
        self.clip_model.eval()

        print('\rloading data', end='')
        transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

        dataset = CocoCaptions(
            root=coco_root,
            annFile=ann_file,
            transform=transform
        )

        self.data_loader = DataLoader(
            dataset=dataset, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, drop_last=True,
            collate_fn=self.collate_fn)

        self.iters = int(len(dataset) / batch_size)

        print('dataloader setup complete')

    def collate_fn(self, data):
        # Returns image_tensor of shape (batch, 3, 256, 256) and
        # text embedding randomly sampled from 5 options of shape (batch, 512)
        images, captions = zip(*data)
        batch_size = len(captions)
        idx = np.random.randint(0, 5, size=(batch_size,))
        captions = [captions[i][idx[i]] for i in range(batch_size)]

        inputs = self.clip_tokenizer(
            captions, padding=True, return_tensors="pt").to(self.device)

        with torch.no_grad():
            text_embeddings = self.clip_model.get_text_features(**inputs)

        images = torch.stack(images).to(self.device)

        return images, text_embeddings


def train_auto_enc(path, epochs=500, lr=1E-3, batch_size=128,
                   device='cpu', img_size=256):
    data = Loader(batch_size=batch_size, img_size=img_size, device=device)

    enc = Encoder(CH=3, latent=32, num_groups=4).to(device)
    dec = Decoder(CH=3, latent=32, num_groups=4).to(device)

    try:
        enc.load_state_dict(torch.load(os.path.join(path, "COCO-enc.pt")))
        dec.load_state_dict(torch.load(os.path.join(path, "COCO-dec.pt")))
    except Exception:
        print('paramerts failed to load from last run')

    optimizer = torch.optim.AdamW(
        list(enc.parameters()) + list(dec.parameters()), lr)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = amp.GradScaler('cuda')

    train_error = []
    MSE = nn.MSELoss()
    MAE = nn.L1Loss()
    # SSIM = ssim_loss()
    KL = KL_Loss()

    dynamic_transforms = transforms.Compose([
        transforms.RandomResizedCrop(
            size=(img_size, img_size), scale=(0.9, 1.1)),
        transforms.RandomHorizontalFlip(p=0.5),
        # transforms.RandomRotation(degrees=5),
        # transforms.ColorJitter(
        #     brightness=0.05, contrast=0.05, saturation=0.05, hue=0.025),
        transforms.RandomGrayscale(p=0.05),
        # transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0)),
        # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    for eps in range(epochs):
        print(f'Epoch {eps + 1}/{epochs}')
        enc.train()
        dec.train()
        running_error = 0.

        lr = optimizer.param_groups[0]['lr']
        print(f'    Learning Rate: {lr}')

        for itr, x in enumerate(tqdm(data.data_loader)):
            optimizer.zero_grad()
            if random.random() > 0.01:
                x = norm(dynamic_transforms(x[0]))
            else:
                x = norm(x[0])

            with amp.autocast('cuda'):
                mu, logvar = enc(x)
                x_ = dec(mu, logvar)

                error = 10 * MSE(x, x_) + \
                    1e-6 * KL(mu, logvar) + 10 * \
                    MAE(x, x_)  # + 2 * SSIM(x, x_)

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
                        num_samples=8, cols=4, dpi=250)

        if (eps + 1) % 1 == 0:
            torch.save(enc.state_dict(), os.path.join(path, "COCO-enc.pt"))
            torch.save(dec.state_dict(), os.path.join(path, "COCO-dec.pt"))

    torch.save(enc.state_dict(), os.path.join(path, "COCO-enc.pt"))
    torch.save(dec.state_dict(), os.path.join(path, "COCO-dec.pt"))


def train(path, epochs=2000, lr=1E-4, batch_size=128,
          steps=1000, emb=64, device='cpu', img_size=512):

    data = Loader(batch_size=batch_size, img_size=img_size, device=device)

    with torch.no_grad():
        enc = Encoder(CH=3, latent=32, num_groups=4).eval().to(device)
        enc.load_state_dict(torch.load(os.path.join(path, "COCO-enc.pt"),
                                       map_location=device))

    dec = Decoder(CH=3, latent=32, num_groups=4).cuda()
    dec.load_state_dict(
        torch.load(os.path.join(path, "COCO-dec.pt")))
    dec.eval()

    model = UNet(in_channels=32, base_channels=32,
                 context_dim=512, time_dim=emb).to(device)
    try:
        model.load_state_dict(torch.load(
            os.path.join(path, "COCO-Autosave-Diffusion.pt")))
    except Exception:
        print('Parameters failed to load from the last diffusion run')

    optimizer = torch.optim.AdamW(model.parameters(), lr)
    # scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, patience=10)
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

    lam_kl = 1E-1

    dynamic_transforms = transforms.Compose([
        transforms.RandomResizedCrop(
            size=(img_size, img_size), scale=(0.9, 1.1)),
        transforms.RandomHorizontalFlip(p=0.5),
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

        if eps + 1 % 50 == 0:
            lam_kl *= 0.1

        for itr, x in enumerate(tqdm(data.data_loader)):
            optimizer.zero_grad()
            if random.random() >= 0.5:
                embdtxt = x[1]
            else:
                embdtxt = None
            if random.random() >= 0.01:
                x = norm(dynamic_transforms(x[0]))
            else:
                x = norm(x[0])
            mu, logvar = enc(x)
            z0 = reparameterize(mu, logvar).detach()
            t = torch.tensor(random.sample(tpop, len(z0)), device=device)
            zt, noise = forward_sample(z0, t, steps, 'lin')
            embdt = embds[t]

            with amp.autocast('cuda'):
                pred_noise = model(zt, embdtxt, embdt)

                kl_error = KL(nn.functional.log_softmax(pred_noise, dim=0),
                              nn.functional.log_softmax(noise, dim=0))

                pix_error = MSE(noise, pred_noise) + MAE(noise, pred_noise)

                error = pix_error + lam_kl * kl_error

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

        scheduler.step(running_error / avg_fact)

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
            # plt.plot(train_error_h, label='Average PixError')
            # plt.plot(train_error_kl, label='Average KL_div')
            plt.legend()
            plt.xlabel("Epochs")
            plt.ylabel("Average Error")
            plt.legend()
            plt.show()

        torch.save(model.state_dict(), os.path.join(
            path, "COCO-Autosave-Diffusion.pt"))


@torch.no_grad()
def fin(path, prompts=[], steps=1000,
        clip_model_name="openai/clip-vit-base-patch32",
        guidance_scale=0.5):
    with torch.no_grad():
        iterations = len(prompts)
        diffusion = Diffusion(steps=steps, scheduler='lin')

        print('\rloading CLIP', end='')
        clip_model = CLIPModel.from_pretrained(clip_model_name)
        clip_tokenizer = CLIPTokenizer.from_pretrained(clip_model_name)
        clip_model.eval()

        inputs = clip_tokenizer(prompts, padding=True, return_tensors="pt")
        text_embeddings = clip_model.get_text_features(**inputs).cuda()

        model = UNet(in_channels=32, base_channels=64,
                     context_dim=512, time_dim=64).cuda()
        try:
            model.load_state_dict(
                torch.load(os.path.join(path, "COCO-Autosave-Diffusion.pt")))
        except Exception:
            print('paramerts failed to load from last diffusion run')

        dec = Decoder(CH=3, latent=32, num_groups=4).cuda()
        dec.load_state_dict(
            torch.load(os.path.join(path, "COCO-dec.pt")))

        dec.eval()
        model.eval()

        idx = torch.linspace(0, steps-1, 16, dtype=torch.int).cuda()
        print(idx)

        z = torch.randn((iterations, 32, 64, 64)).cuda()
        z_min, z_max = z.min().item(), z.max().item()
        print(z_min, z_max)

        imgs = []

        for t in trange(0, steps):
            z_c = torch.clamp(diffusion.backward(
                z, torch.tensor(steps - 1 - t), model,
                text_embeddings), -5.5, 5.5)
            z_uc = torch.clamp(diffusion.backward(
                z, torch.tensor(steps - 1 - t), model,
                None), -5.5, 5.5)
            z = z_uc + guidance_scale * (z_c - z_uc)
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
    path = os.path.abspath(__file__)[:-45]
    os.chdir(path)
    print(path)

    prompts = [
        "A red car driving by a green hill on a rainy day",
        "A cute cat sitting by a delicious pizza",
        "A plane flying into the clouds above the mountains",
        "A toilet in the desert full of colorful lights",
        # "A person holding a smartphone in a red case.",
        # "A young woman heads a soccer ball as others watch.",
        # "A man in sunglasses is hitting a ball with a racket.",
        # "A half-eaten sandwich, waffle fries, and soda pop.",
        # "A robot painting on a canvas in a futuristic city",
        # "A giraffe wearing sunglasses at a beach party",
        # "A dog riding a skateboard in a busy city street",
        # "A wizard casting spells in a forest full of fireflies",
        # "A snowman surfing on a wave during sunset",
        # "A unicorn sitting under a rainbow in a field of flowers",
        # "A futuristic motorcycle racing through neon streets",
        # "A coffee cup with a tiny forest growing inside it",
        # "A giant rubber duck floating in the middle of a city",
        # "A squirrel holding an umbrella in a rainstorm",
        # "A haunted house with pumpkins and glowing windows",
        # "A panda playing a piano under a spotlight",
        # "A city skyline with flying cars and glowing billboards",
        # "A ballerina dancing on top of a skyscraper",
        # "A chef juggling vegetables in a busy kitchen",
        # "A polar bear wearing sunglasses and drinking a soda",
        # "A spaceship landing in the middle of a desert",
        # "A mermaid sitting on a rock with a city skyline behind",
        # "A lion with a crown sitting on a throne in a jungle",
        # "A waterfall flowing into a river of glowing blue water",
        # "A garden gnome riding a dragon through a castle",
        # "A donut shop shaped like a giant donut at night",
        # "A fox with a scarf reading a book under a tree",
        # "A moonlit beach with glowing jellyfish in the waves",
        # "A hot air balloon made of candy floating in the sky",
        # "A treehouse built on a giant mushroom in a fantasy forest",
        # "A dolphin jumping through colorful, glowing rings in the ocean",
        # "A friendly robot serving coffee at a cozy cafe",
        # "A treasure chest filled with glowing gems underwater",
        # "A n16eon-lit diner in the middle of a desert at night",
        # "A vintage car covered in vibrant graffiti parked by a mural",
        # "A giant octopus hugging a lighthouse during sunset",
        # "A fairy reading a book by a small waterfall",
        # "A skatepark on the moon with Earth visible in the background",
        # "A superhero flying through a futuristic city with tall buildings",
        # "A wolf howling under a full moon surrounded by fog",
        # "A child holding a glowing balloon in a dark forest",
        # "A futuristic lab with robots working on floating screens",
        # "A desert oasis with crystal-clear water and palm trees",
        # "A dragon breathing fire over a snowy mountain range",
        # "A whale swimming through clouds in a surreal sky",
        # "A picnic setup on a small island in the middle of a lake",
        # "A train traveling across a bridge above a cloud sea",
        # "A fox standing on a cliff looking over a misty valley",
        # "A steampunk airship flying over a Victorian city",
        # "A penguin wearing a top hat sliding on ice",
        # "A bustling marketplace with colorful fabrics and spices",
        # "A peacock spreading its feathers in a magical forest",
        # "A robot dog running through a park with other dogs",
        # "A river running through a valley with bioluminescent plants",
        # "A camel caravan traveling under a sky filled with stars",
        # "A beautiful palace built on top of a mountain peak",
        # "A butterfly with wings made of stained glass in a garden",
        # "A cityscape where every building is shaped like an animal",
        # "A group of kids playing in a magical treehouse village",
        # "A chef cooking on a floating kitchen platform in the sky",
    ]

    itr = len(prompts)
    img_size = 512
    steps = 1000

    a = input('Train model from last checkpoint?(y/n)')
    if a == 'y':
        a = input('Train autoencoder(a) or diffusion(d)?(a/d)')
        if a == 'a':
            train_auto_enc(path=path, device='cuda',
                           batch_size=64, epochs=1000, lr=3.14E-5,
                           img_size=img_size)
        else:
            train(path=path, epochs=1000, img_size=img_size, lr=1E-3,
                  batch_size=32, steps=steps, emb=64, device='cuda')

    else:
        with torch.no_grad():
            for _ in range(1):
                y, zy = fin(path, prompts=prompts, steps=steps)

            if True:
                data = Loader(batch_size=itr, img_size=img_size,
                              device='cpu')
                enc = Encoder(CH=3, latent=32, num_groups=4)
                enc.load_state_dict(torch.load(os.path.join(path,
                                                            "COCO-enc.pt"),
                                               map_location='cpu'))
                dec = Decoder(CH=3, latent=32, num_groups=4)
                dec.load_state_dict(torch.load(os.path.join(path,
                                                            "COCO-dec.pt"),
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
                show_images(torch.abs(z.mean(dim=0) - zy.mean(dim=0)
                                      ).detach().cpu().unsqueeze(dim=1),
                            zy.shape[1], int(zy.shape[1] ** 0.5))
