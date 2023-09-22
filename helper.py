"""
Created on Wed Aug 23 19:21:10 2023

@author: Agam Chopra
"""
import torch
import matplotlib.pyplot as plt


def norm(x):
    return (x - x.min()) / (x.max() - x.min())


def show_images(data, num_samples=9, cols=3, mode=True,
                size=(15, 15), dpi=500):
    data = (data - data.min()) / (data.max() - data.min())
    plt.figure(figsize=size, dpi=dpi)
    for i, img in enumerate(data):
        if i == num_samples:
            break
        plt.subplot(int(num_samples/cols) + 1, cols, i + 1)
        plt.axis('off')
        if img.shape[0] == 1:
            plt.imshow(img[0], cmap='magma')
        else:
            if mode:
                plt.imshow(img.permute(1, 2, 0))
            else:
                plt.imshow(img.permute(2, 1, 0))
    plt.show()


def distributions(x, y):
    plt.figure(figsize=(10, 5), dpi=500)

    idx = torch.linspace(-4.7, 4.7, 256)
    noise = torch.randn((x.shape))

    d0 = torch.histc(noise, bins=256, min=-4.7, max=4.7)
    d1 = torch.histc(x, bins=256, min=-4.7, max=4.7)
    d2 = torch.histc(y, bins=256, min=-4.7, max=4.7)

    d0 /= d0.max()
    d1 /= d1.max()
    d2 /= d2.max()

    plt.plot(idx, d0, 'k-', label='Noise')
    plt.plot(idx, d1, 'm-', label='Target')
    plt.plot(idx, d2, 'r-', label='Predicted')
    plt.legend()
    plt.title(f'n = {x.shape[0]}')
    plt.show()


def getPositionEncoding(seq_len, d=64, n=10000):
    P = torch.zeros((seq_len, d))
    for k in range(seq_len):
        for i in torch.arange(int(d/2)):
            denominator = torch.pow(n, 2*i/d)
            P[k, 2*i] = torch.sin(k/denominator)
            P[k, 2*i+1] = torch.cos(k/denominator)
    return P


def get_cos_betas(steps, max_beta=0.999):
    def alpha_bar(t): return torch.cos((t + 0.008) / 1.008 * torch.pi / 2) ** 2
    i = torch.linspace(0, steps - 1, steps)
    t1 = i / steps
    t2 = (i + 1) / steps
    betas = 1 - alpha_bar(t2) / alpha_bar(t1)
    betas_clipped = torch.clamp(betas, 0., max_beta)
    return betas_clipped


def get_betas(steps=1000, scheduler='lin'):
    if scheduler == 'lin':
        scale = 1000 / steps
        start = scale * 0.0001
        end = scale * 0.02
        return torch.linspace(start, end, steps)
    elif scheduler == 'cos':
        return get_cos_betas(steps)
    else:
        raise NotImplementedError(f"scheduler not implemented: {scheduler}")


def forward_sample(x0, t, steps, scheduler='lin'):
    betas = get_betas(steps, scheduler).to(x0.device)
    alphas = 1 - betas
    alpha_hat = torch.cumprod(alphas, dim=0)
    alpha_hat_t = torch.gather(alpha_hat, dim=-1,
                               index=t.to(x0.device)).view(-1, 1, 1, 1)
    noise = torch.randn_like(x0, device=x0.device)
    mean = alpha_hat_t.sqrt() * x0
    var = torch.sqrt(1 - alpha_hat_t) * noise
    xt = mean + var
    return xt, noise


class Diffusion:
    def __init__(self, start=0.0001, end=0.02, steps=1000,
                 emb=64, scheduler='lin'):
        self.start = start
        self.end = end
        self.steps = steps
        self.beta = get_betas(steps, scheduler)
        self.alpha = 1 - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)
        self.embeddings = getPositionEncoding(steps, d=emb, n=10000)

    @torch.no_grad()
    def backward(self, x, t, model):
        t = t.to(x.device)
        self.beta = self.beta.to(x.device)
        self.alpha = self.alpha.to(x.device)
        self.alpha_hat = self.alpha_hat.to(x.device)
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
