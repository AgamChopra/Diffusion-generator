"""
Created on Wed Aug 23 19:21:10 2023

@author: Agam Chopra
"""
import torch
import matplotlib.pyplot as plt


def show_images(data, num_samples=9, cols=3):
    plt.figure(figsize=(15, 15))
    for i, img in enumerate(data):
        if i == num_samples:
            break
        plt.subplot(int(num_samples/cols) + 1, cols, i + 1)
        plt.axis('off')
        if img.shape[0] == 1:
            plt.imshow(img[0], cmap='magma')
        else:
            plt.imshow(img.permute(1, 2, 0))
    plt.show()


def getPositionEncoding(seq_len, d=64, n=10000):
    P = torch.zeros((seq_len, d))
    for k in range(seq_len):
        for i in torch.arange(int(d/2)):
            denominator = torch.pow(n, 2*i/d)
            P[k, 2*i] = torch.sin(k/denominator)
            P[k, 2*i+1] = torch.cos(k/denominator)
    return P


def forward_sample(x0, t, steps, start=0., end=1.):
    betas = torch.linspace(start, end, steps).to(x0.device)
    alphas = 1 - betas
    alpha_hat = torch.cumprod(alphas, dim=0)
    alpha_hat_t = torch.gather(alpha_hat, dim=-1,
                               index=t.to(x0.device)).view(-1, 1, 1, 1)
    noise = torch.randn_like(x0).to(x0.device)
    mean = alpha_hat_t.sqrt() * x0
    var = torch.sqrt(1 - alpha_hat_t) * noise
    xt = mean + var
    return xt, noise


class Diffusion:
    def __init__(self, start=0.0001, end=0.02, steps=1000, emb=64):
        self.start = start
        self.end = end
        self.steps = steps
        self.beta = torch.linspace(start, end, steps)
        self.alpha = 1 - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)
        self.embeddings = getPositionEncoding(steps, d=emb, n=10000)

    @torch.no_grad()
    def backward(self, x, t, model):
        embeddings_t = self.embeddings[t].to(x.device)
        beta_t = torch.gather(self.beta, dim=-1,
                              index=t).view(-1, 1, 1, 1).to(x.device)
        sqrt_one_minus_alpha_hat_t = torch.gather(torch.sqrt(
            1. - self.alpha_hat), dim=-1,
            index=t).view(-1, 1, 1, 1).to(x.device)
        sqrt_inv_alpha_t = torch.gather(torch.sqrt(
            1.0 / self.alpha), dim=-1, index=t).view(-1, 1, 1, 1).to(x.device)
        mean = sqrt_inv_alpha_t * \
            (x - beta_t * model(x, embeddings_t) / sqrt_one_minus_alpha_hat_t)
        posterior_variance_t = beta_t

        if t == 0:
            return mean
        else:
            noise = torch.randn_like(x, device=x.device)
            varience = torch.sqrt(posterior_variance_t) * noise
            return mean + varience
