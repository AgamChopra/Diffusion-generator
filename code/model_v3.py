import torch
import torch.nn as nn

from model_v2 import pad2d, MultiKernelConv2d


class Block(nn.Module):
    def __init__(self, in_c, text_dim, time_dim, out_c, hid_c=None, num_groups=8):
        super(Block, self).__init__()

        if hid_c is None:
            self.mlp_time = nn.Sequential(
                nn.Linear(time_dim, out_c), nn.Mish())
            self.mlp_text = nn.Sequential(
                nn.Linear(text_dim, out_c), nn.Mish())

            self.layer = nn.Sequential(
                MultiKernelConv2d(in_channels=in_c, out_channels=out_c),
                nn.Mish(),
                nn.GroupNorm(num_groups, out_c)
            )

            self.out_block = nn.Sequential(
                MultiKernelConv2d(in_channels=out_c, out_channels=out_c),
                nn.Mish(),
                nn.GroupNorm(num_groups, out_c)
            )
        else:
            self.mlp_time = nn.Sequential(
                nn.Linear(time_dim, hid_c), nn.Mish())
            self.mlp_text = nn.Sequential(
                nn.Linear(text_dim, hid_c), nn.Mish())

            self.layer = nn.Sequential(
                MultiKernelConv2d(in_channels=in_c, out_channels=hid_c),
                nn.Mish(),
                nn.GroupNorm(num_groups, hid_c)
            )

            self.out_block = nn.Sequential(
                MultiKernelConv2d(in_channels=hid_c, out_channels=hid_c),
                nn.Mish(),
                nn.GroupNorm(num_groups, hid_c),
                nn.Upsample(scale_factor=2, mode='bilinear',
                            align_corners=True),
                MultiKernelConv2d(in_channels=hid_c, out_channels=out_c),
                nn.Mish(),
                nn.GroupNorm(num_groups, out_c)
            )

    def forward(self, x, txt, t):
        t = self.mlp_time(t)
        txt = self.mlp_text(txt)

        y = self.layer(x)

        t = t[(..., ) + (None, ) * 2]
        txt = txt[(..., ) + (None, ) * 2]

        y = y + t + txt
        y = self.out_block(y)
        return y


class UNet(nn.Module):
    def __init__(self, in_channels=256, time_dim=64, context_dim=512,
                 base_channels=64, num_groups=4):
        super(UNet, self).__init__()

        self.time_mlp = nn.Sequential(nn.Linear(time_dim, time_dim), nn.Mish())
        self.text_mlp = nn.Sequential(
            nn.Linear(context_dim, context_dim), nn.Mish())

        self.layer1 = nn.Sequential(
            MultiKernelConv2d(in_channels, base_channels),
            nn.Mish(),
            nn.GroupNorm(num_groups, base_channels)
        )

        self.layer2 = Block(in_c=base_channels, text_dim=context_dim, time_dim=time_dim,
                            out_c=base_channels*2, num_groups=num_groups)

        self.layer5 = Block(in_c=base_channels*2, text_dim=context_dim, time_dim=time_dim,
                            out_c=base_channels*2, hid_c=base_channels*4, num_groups=num_groups)

        self.layer8 = Block(in_c=base_channels*4, text_dim=context_dim, time_dim=time_dim,
                            out_c=base_channels*2, num_groups=num_groups)

        self.out = nn.Sequential(
            nn.Conv2d(in_channels=base_channels*2,
                      out_channels=in_channels, kernel_size=1)
        )

        self.pool2 = nn.Sequential(
            nn.Conv2d(in_channels=base_channels*2,
                      out_channels=base_channels*2, kernel_size=2, stride=2),
            nn.Mish(),
            nn.GroupNorm(num_groups, base_channels*2)
        )

    def forward(self, x, txt, t):
        t = self.time_mlp(t)
        txt = self.text_mlp(txt)

        y = self.layer1(x)

        y2 = self.layer2(y, txt, t)
        y = self.pool2(y2)

        y = self.layer5(y, txt, t)

        y = torch.cat((y2, pad2d(y, y2)), dim=1)
        y = self.layer8(y, txt, t)
        y = pad2d(y, x)

        y = self.out(y)

        return y


# Example usage
if __name__ == "__main__":
    batch_size = 16
    image = torch.randn(batch_size, 32, 64, 64)
    text_embedding = torch.randn(batch_size, 512)
    time_embedding = torch.randn(batch_size, 64)

    model = UNet(in_channels=32, base_channels=64,
                 context_dim=512, time_dim=64)
    output = model(image, text_embedding, time_embedding)
    print(output.shape)  # Should output: torch.Size([16, 32, 64, 64])
