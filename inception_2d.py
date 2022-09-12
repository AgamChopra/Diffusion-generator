import torch.nn as nn
import torch


INCEP_C = 16
FACTOR = 2


class Inception(nn.Module):
    def __init__(self, c_in):
        super(Inception, self).__init__()
        C = INCEP_C*FACTOR

        self.F1 = nn.Sequential(nn.Conv2d(c_in, C*2, kernel_size=1, stride=1),
                                nn.InstanceNorm2d(C*2), nn.LeakyReLU(inplace=False))

        self.F2 = nn.Sequential(nn.Conv2d(c_in, C*3, kernel_size=1, stride=1, padding=1),
                                nn.InstanceNorm2d(
                                    C*3), nn.LeakyReLU(inplace=False),
                                nn.Conv2d(C*3, C*4, kernel_size=3, stride=1),
                                nn.InstanceNorm2d(C*4), nn.LeakyReLU(inplace=False))

        self.F3 = nn.Sequential(nn.Conv2d(c_in, C, kernel_size=1, stride=1, padding=2),
                                nn.InstanceNorm2d(C), nn.LeakyReLU(
                                    inplace=False),
                                nn.Conv2d(C, C*1, kernel_size=5, stride=1),
                                nn.InstanceNorm2d(C*1), nn.LeakyReLU(inplace=False))

        self.F4 = nn.Sequential(nn.ConstantPad2d(1, 0.), nn.MaxPool2d(kernel_size=3, stride=1),
                                nn.Conv2d(c_in, C*1, kernel_size=1, stride=1),
                                nn.InstanceNorm2d(C*1), nn.LeakyReLU(inplace=False))

    def forward(self, x):
        y = torch.cat((self.F1(x), self.F2(x), self.F3(x), self.F4(x)), dim=1)
        return y


class Inception_model(nn.Module):
    def __init__(self, C=3):
        super(Inception_model, self).__init__()

        self.L0 = nn.Sequential(Inception(C),
                                Inception(8*INCEP_C*FACTOR),
                                Inception(8*INCEP_C*FACTOR))

        self.L1 = nn.Sequential(Inception(8*INCEP_C*FACTOR),
                                Inception(8*INCEP_C*FACTOR),
                                Inception(8*INCEP_C*FACTOR))

        self.L2 = nn.Sequential(Inception(2*8*INCEP_C*FACTOR),
                                Inception(8*INCEP_C*FACTOR),
                                Inception(8*INCEP_C*FACTOR))

        self.L3 = nn.Sequential(Inception(2*8*INCEP_C*FACTOR),
                                Inception(8*INCEP_C*FACTOR),
                                Inception(8*INCEP_C*FACTOR))

        self.L4 = nn.Sequential(Inception(2*8*INCEP_C*FACTOR),
                                Inception(8*INCEP_C*FACTOR),
                                Inception(8*INCEP_C*FACTOR))

        self.L5 = nn.Sequential(Inception(2*8*INCEP_C*FACTOR),
                                Inception(8*INCEP_C*FACTOR),
                                Inception(8*INCEP_C*FACTOR))

        self.L_out = nn.Sequential(Inception(2*8*INCEP_C*FACTOR),
                                   nn.Conv2d(8*INCEP_C*FACTOR, C, kernel_size=1, stride=1))

    def forward(self, x):
        y0 = self.L0(x)

        y1 = self.L1(y0)
        x = torch.cat((y1, y0), dim=1)

        y2 = self.L2(x)
        x = torch.cat((y2, y1), dim=1)

        y3 = self.L3(x)
        x = torch.cat((y3, y2), dim=1)

        y4 = self.L4(x)
        x = torch.cat((y4, y3), dim=1)

        y = self.L_out(x)
        return y


class Inception_v2(nn.Module):
    def __init__(self, C=3, device='cpu'):
        super(Inception_v2, self).__init__()
        self.P = Inception_model(C).to(device)

    def forward(self, x):
        y = self.P(x)
        return y


def main(x, y):
    M = Inception_v2(3, 'cuda')

    x = torch.rand((8, 3, x, y))
    x = (x - x.min())/(x.max() - x.min())
    x = x.to(device='cuda')

    y = M(x)
    print(M)
    print(y.shape, y.device)


if __name__ == "__main__":
    x, y = 64, 64
    main(x, y)
