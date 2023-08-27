import torch
import torch.nn as nn
from math import ceil


def pad2d(input_, target):  # pads if target is bigger and crops if target is smaller
    delta = [target.shape[2+i] - input_.shape[2+i] for i in range(2)]
    output = nn.functional.pad(input=input_,
                               pad=(ceil(delta[1]/2),
                                    delta[1] - ceil(delta[1]/2),
                                    ceil(delta[0]/2),
                                    delta[0] - ceil(delta[0]/2)),
                               mode='constant',
                               value=0).to(dtype=torch.float,
                                           device=input_.device)
    return output


class Block(nn.Module):
    def __init__(self, in_c, embd_dim, out_c, hid_c=None):
        super(Block, self).__init__()

        if hid_c is None:
            self.mlp = nn.Sequential(nn.Linear(embd_dim, out_c), nn.ReLU())

            self.layer = nn.Sequential(nn.Conv2d(
                in_channels=in_c, out_channels=out_c, kernel_size=3), nn.ReLU(), nn.BatchNorm2d(out_c))

            self.out_block = nn.Sequential(nn.Conv2d(
                in_channels=out_c, out_channels=out_c, kernel_size=2), nn.ReLU(), nn.BatchNorm2d(out_c))
        else:
            self.mlp = nn.Sequential(nn.Linear(embd_dim, hid_c), nn.ReLU())

            self.layer = nn.Sequential(nn.Conv2d(
                in_channels=in_c, out_channels=hid_c, kernel_size=3), nn.ReLU(), nn.BatchNorm2d(hid_c))

            self.out_block = nn.Sequential(nn.Conv2d(in_channels=hid_c, out_channels=hid_c, kernel_size=2), nn.ReLU(), nn.BatchNorm2d(hid_c),
                                           nn.ConvTranspose2d(in_channels=hid_c, out_channels=out_c, kernel_size=2, stride=2), nn.ReLU(), nn.BatchNorm2d(out_c))

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
            nn.Conv2d(CH, int(64/n), 2, 1), nn.ReLU(), nn.BatchNorm2d(int(64/n)))

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

        self.pool2 = nn.Conv2d(in_channels=int(
            128/n), out_channels=int(128/n), kernel_size=2, stride=2)

        self.pool3 = nn.Conv2d(in_channels=int(
            256/n), out_channels=int(256/n), kernel_size=2, stride=2)

        self.pool4 = nn.Conv2d(in_channels=int(
            512/n), out_channels=int(512/n), kernel_size=2, stride=2)

    def forward(self, x, t):
        t = self.time_mlp(t)

        x_pad = pad2d(x, torch.ones((1, 1, 65, 65))) if x.shape[2] < 65 else x

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


def test(device='cpu'):
    batch = 1
    a = torch.ones((batch, 1, 64, 64), device=device)
    t = torch.ones((batch, 32), device=device)

    model = UNet(CH=1, n=64).to(device)
    print(model)

    b = model(a, t, device)

    print(a.shape, t.shape)
    print(b.shape)

    from matplotlib import pyplot as plt

    plt.imshow(a[0].T.detach().cpu().numpy())
    plt.show()

    plt.imshow(b[0].T.detach().cpu().numpy())
    plt.show()


if __name__ == '__main__':
    test('cuda:0')
