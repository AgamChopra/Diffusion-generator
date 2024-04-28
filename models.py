import torch
import torch.nn as nn
from math import ceil


def pad2d(inpt, target):  # pads if target is bigger and crops if smaller
    if torch.is_tensor(target):
        delta = [target.shape[2+i] - inpt.shape[2+i] for i in range(2)]
    else:
        delta = [target[i] - inpt.shape[2+i] for i in range(2)]
    output = nn.functional.pad(input=inpt,
                               pad=(ceil(delta[1]/2),
                                    delta[1] - ceil(delta[1]/2),
                                    ceil(delta[0]/2),
                                    delta[0] - ceil(delta[0]/2)),
                               mode='constant',
                               value=0).to(dtype=torch.float,
                                           device=inpt.device)
    return output


class Block(nn.Module):
    def __init__(self, in_c, embd_dim, out_c, hid_c=None):
        super(Block, self).__init__()

        if hid_c is None:
            self.mlp = nn.Sequential(nn.Linear(embd_dim, out_c), nn.ReLU())

            self.layer = nn.Sequential(nn.Conv2d(
                in_channels=in_c, out_channels=out_c, kernel_size=3),
                nn.ReLU(), nn.BatchNorm2d(out_c))

            self.out_block = nn.Sequential(nn.Conv2d(
                in_channels=out_c, out_channels=out_c, kernel_size=2),
                nn.ReLU(), nn.BatchNorm2d(out_c))
        else:
            self.mlp = nn.Sequential(nn.Linear(embd_dim, hid_c), nn.ReLU())

            self.layer = nn.Sequential(nn.Conv2d(
                in_channels=in_c, out_channels=hid_c, kernel_size=3),
                nn.ReLU(), nn.BatchNorm2d(hid_c))

            self.out_block = nn.Sequential(nn.Conv2d(in_channels=hid_c,
                                                     out_channels=hid_c,
                                                     kernel_size=2),
                                           nn.ReLU(), nn.BatchNorm2d(hid_c),
                                           nn.ConvTranspose2d(in_channels=hid_c,
                                                              out_channels=out_c,
                                                              kernel_size=2,
                                                              stride=2),
                                           nn.ReLU(), nn.BatchNorm2d(out_c))

    def forward(self, x, t):
        t = self.mlp(t)
        y = self.layer(x)
        t = t[(..., ) + (None, ) * 2]
        y = y + t
        y = self.out_block(y)
        return y


class BlockL(nn.Module):
    def __init__(self, in_c, embd_dim, out_c, hid_c=None):
        super(BlockL, self).__init__()

        if hid_c is None:
            self.mlp = nn.Sequential(
                nn.Linear(embd_dim, int(out_c / 4)), nn.ReLU())

            self.layer = nn.Sequential(nn.Conv2d(
                in_channels=in_c, out_channels=int(out_c / 4), kernel_size=2),
                nn.ReLU(), nn.BatchNorm2d(int(out_c / 4)))

            self.out_block = nn.Sequential(nn.Conv2d(
                in_channels=int(out_c / 4), out_channels=int(out_c / 2),
                kernel_size=2), nn.ReLU(), nn.BatchNorm2d(int(out_c / 2)),
                nn.Conv2d(in_channels=int(out_c / 2), out_channels=out_c,
                          kernel_size=2), nn.ReLU(), nn.BatchNorm2d(out_c))
        else:
            self.mlp = nn.Sequential(nn.Linear(embd_dim, hid_c), nn.ReLU())

            self.layer = nn.Sequential(nn.Conv2d(
                in_channels=in_c, out_channels=hid_c, kernel_size=2),
                nn.ReLU(), nn.BatchNorm2d(hid_c))

            self.out_block = nn.Sequential(nn.Conv2d(in_channels=hid_c,
                                                     out_channels=2*hid_c,
                                                     kernel_size=2),
                                           nn.ReLU(), nn.BatchNorm2d(2*hid_c),
                                           nn.Conv2d(in_channels=2*hid_c,
                                                     out_channels=4*hid_c,
                                                     kernel_size=2),
                                           nn.ReLU(), nn.BatchNorm2d(4*hid_c),
                                           nn.ConvTranspose2d(in_channels=4*hid_c,
                                                              out_channels=out_c,
                                                              kernel_size=2,
                                                              stride=2),
                                           nn.ReLU(), nn.BatchNorm2d(out_c))

    def forward(self, x, t):
        t = self.mlp(t)
        y = self.layer(x)
        t = t[(..., ) + (None, ) * 2]
        y = y + t
        y = self.out_block(y)
        return y


class UNet(nn.Module):
    def __init__(self, CH=1, emb=64, n=1):
        super(UNet, self).__init__()
        # layers
        self.time_mlp = nn.Sequential(nn.Linear(emb, emb), nn.ReLU())

        self.layer1 = nn.Sequential(
            nn.Conv2d(CH, int(64/n), 2, 1), nn.ReLU(),
            nn.BatchNorm2d(int(64/n)))

        self.layer2 = Block(in_c=int(64/n), embd_dim=emb, out_c=int(128/n))

        self.layer3 = Block(in_c=int(128/n), embd_dim=emb, out_c=int(256/n))

        self.layer4 = Block(in_c=int(256/n), embd_dim=emb, out_c=int(512/n))

        self.layer5 = Block(in_c=int(512/n), embd_dim=emb,
                            out_c=int(512/n), hid_c=int(1024/n))

        self.layer6 = Block(in_c=int(1024/n), embd_dim=emb,
                            out_c=int(256/n), hid_c=int(512/n))

        self.layer7 = Block(in_c=int(512/n), embd_dim=emb,
                            out_c=int(128/n), hid_c=int(256/n))

        self.layer8 = Block(in_c=int(256/n), embd_dim=emb, out_c=int(64/n))

        self.out = nn.Sequential(nn.Conv2d(in_channels=int(64/n),
                                           out_channels=int(64/n),
                                           kernel_size=1),
                                 nn.ReLU(), nn.BatchNorm2d(int(64/n)),
                                 nn.Conv2d(in_channels=int(64/n),
                                           out_channels=CH, kernel_size=1))

        self.pool2 = nn.Sequential(nn.Conv2d(in_channels=int(
            128/n), out_channels=int(128/n), kernel_size=2, stride=2),
            nn.ReLU(), nn.BatchNorm2d(int(128/n)))

        self.pool3 = nn.Sequential(nn.Conv2d(in_channels=int(
            256/n), out_channels=int(256/n), kernel_size=2, stride=2),
            nn.ReLU(), nn.BatchNorm2d(int(256/n)))

        self.pool4 = nn.Sequential(nn.Conv2d(in_channels=int(
            512/n), out_channels=int(512/n), kernel_size=2, stride=2),
            nn.ReLU(), nn.BatchNorm2d(int(512/n)))

    def forward(self, x, t):
        t = self.time_mlp(t)

        if x.shape[2] < 64:
            x_pad = pad2d(x, torch.ones((1, 1, 65, 65)))
        elif x.shape[2] < 71:
            x_pad = pad2d(x, torch.ones((1, 1, 71, 71)))
        elif x.shape[2] < 135:
            x_pad = pad2d(x, torch.ones((1, 1, 135, 135)))
        else:
            x_pad = x

        y = self.layer1(x_pad)

        y2 = self.layer2(y, t)
        y = self.pool2(y2)

        y3 = self.layer3(y, t)
        y = self.pool3(y3)

        y4 = self.layer4(y, t)
        y = self.pool4(y4)

        y = self.layer5(y, t)

        y = torch.cat((y4, pad2d(y, y4)), dim=1)
        y = self.layer6(y, t)

        y = torch.cat((y3, pad2d(y, y3)), dim=1)
        y = self.layer7(y, t)

        y = torch.cat((y2, pad2d(y, y2)), dim=1)
        y = self.layer8(y, t)

        y = pad2d(y, x)

        y = self.out(y)

        return y


class Encoder(nn.Module):
    def __init__(self, CH=1, latent=128):
        super(Encoder, self).__init__()
        self.layer1 = nn.Sequential(nn.Conv2d(
            in_channels=CH, out_channels=int(latent/3), kernel_size=2),
            nn.ReLU(), nn.BatchNorm2d(int(latent/3)),
            nn.Conv2d(in_channels=int(latent/3),
                      out_channels=int(latent/3), kernel_size=2, stride=2),
            nn.ReLU(), nn.BatchNorm2d(int(latent/3)))

        self.layer2 = nn.Sequential(nn.Conv2d(
            in_channels=int(latent/3), out_channels=int(latent/2),
            kernel_size=2),
            nn.ReLU(), nn.BatchNorm2d(int(latent/2)),
            nn.Conv2d(in_channels=int(latent/2),
                      out_channels=int(latent/2), kernel_size=2, stride=2),
            nn.ReLU(), nn.BatchNorm2d(int(latent/2)))

        self.layer3 = nn.Sequential(nn.Conv2d(
            in_channels=int(latent/2), out_channels=latent, kernel_size=2),
            nn.ReLU(), nn.BatchNorm2d(latent),
            nn.Conv2d(in_channels=latent, out_channels=latent,
                      kernel_size=2, stride=2),
            nn.ReLU(), nn.BatchNorm2d(latent))

    def forward(self, x):
        y = self.layer3(self.layer2(self.layer1(x)))
        return y


class Decoder(nn.Module):
    def __init__(self, CH=1, latent=128):
        super(Decoder, self).__init__()
        self.layer1 = nn.Sequential(nn.Conv2d(
            in_channels=latent, out_channels=latent, kernel_size=2, padding=2),
            nn.ReLU(), nn.BatchNorm2d(latent),
            nn.ConvTranspose2d(
            in_channels=latent, out_channels=int(latent/2), kernel_size=2,
            stride=2, padding=1),
            nn.ReLU(), nn.BatchNorm2d(int(latent/2)))

        self.layer2 = nn.Sequential(nn.Conv2d(
            in_channels=int(latent/2), out_channels=int(latent/2),
            kernel_size=2, padding=0),
            nn.ReLU(), nn.BatchNorm2d(int(latent/2)),
            nn.ConvTranspose2d(in_channels=int(latent/2), out_channels=int(
                latent/3), kernel_size=2, stride=2, padding=0),
            nn.ReLU(), nn.BatchNorm2d(int(latent/3)))

        self.layer3 = nn.Sequential(nn.Conv2d(
            in_channels=int(latent/3), out_channels=int(latent/3),
            kernel_size=2, padding=0),
            nn.ReLU(), nn.BatchNorm2d(int(latent/3)),
            nn.ConvTranspose2d(in_channels=int(
                latent/3), out_channels=int(latent/4), kernel_size=2,
                stride=2, padding=0),
            nn.ReLU(), nn.BatchNorm2d(int(latent/4)),
            nn.Conv2d(in_channels=int(latent/4), out_channels=CH,
                      kernel_size=3, padding=0))

    def forward(self, x):
        y = self.layer3(self.layer2(self.layer1(x)))
        return y


def test_unet(device='cpu'):
    batch = 1
    a = torch.randn((batch, 3, 128, 128), device=device)
    t = torch.rand((batch, 64), device=device)

    model = UNet(CH=3, n=64).to(device)

    b = model(a, t)

    print(a.shape, t.shape)
    print(b.shape)

    a = (a - a.min())/(a.max() - a.min())
    b = (b - b.min())/(b.max() - b.min())

    from matplotlib import pyplot as plt

    plt.imshow(a[0].T.detach().cpu().numpy())
    plt.show()

    plt.imshow(b[0].T.detach().cpu().numpy())
    plt.show()


def test_latent(device='cpu'):
    batch = 1
    x = torch.randn((batch, 3, 128, 128), device=device)
    t = torch.rand((batch, 64), device=device)
    print(x.shape, t.shape)

    enc = Encoder(CH=3, latent=4).to(device)
    dec = Decoder(CH=3, latent=4).to(device)
    lat = UNet(CH=4, n=64).to(device)

    z = enc(x)
    print('z shape:', z.shape)

    z_t_p = lat(z, t)
    print('zp shape:', z_t_p.shape)
    print('shape z matches z_t_p:', z.shape == z_t_p.shape)

    x_p = dec(z)
    print(x_p.shape)

    from matplotlib import pyplot as plt
    from helper import norm

    plt.imshow(norm(x)[0].T.detach().cpu().numpy())
    plt.show()

    plt.imshow(norm(x_p)[0].T.detach().cpu().numpy())
    plt.show()

    x_p = dec(z_t_p)
    print(x_p.shape)

    plt.imshow(norm(x_p)[0].T.detach().cpu().numpy())
    plt.show()


if __name__ == '__main__':
    # test_unet('cuda:0')
    test_latent('cuda:0')
