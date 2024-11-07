import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossAttention(nn.Module):
    def __init__(self, dim, context_dim):
        super(CrossAttention, self).__init__()
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(context_dim, dim)
        self.value = nn.Linear(context_dim, dim)
        self.scale = dim ** -0.5

    def forward(self, x, context):
        # Ensure that the context has the same batch size as x
        if context.dim() == 2:
            context = context.unsqueeze(1)

        q = self.query(x)
        k = self.key(context)
        v = self.value(context)

        attn_scores = torch.einsum('bqd,bkd->bqk', q, k) * self.scale
        attn_weights = F.softmax(attn_scores, dim=-1)
        attended_values = torch.einsum('bqk,bkd->bqd', attn_weights, v)

        return attended_values


class UNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, context_dim):
        super(UNetBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels,
                               kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels,
                               kernel_size=3, padding=1)
        self.cross_attn = CrossAttention(out_channels, context_dim)
        self.norm = nn.GroupNorm(8, out_channels)
        self.activation = nn.Mish()

    def forward(self, x, context):
        x = self.conv1(x)
        x = self.activation(self.norm(x))
        x = self.conv2(x)
        x = self.activation(self.norm(x))

        # Cross Attention over the spatial dimensions
        b, c, h, w = x.shape
        x_reshaped = x.view(b, c, -1).permute(0, 2, 1)  # (b, h*w, c)
        attn_out = self.cross_attn(x_reshaped, context)
        attn_out = attn_out.permute(0, 2, 1).view(b, c, h, w)  # (b, c, h, w)

        return x + attn_out


class UNet(nn.Module):
    def __init__(self, in_channels=3, base_channels=64,
                 context_dim=512, time_dim=64):
        super(UNet, self).__init__()
        self.init_conv = nn.Conv2d(
            in_channels, base_channels, kernel_size=3, padding=1,
            padding_mode='reflect')

        # Downsampling blocks
        self.down1 = UNetBlock(
            base_channels, base_channels * 2, context_dim + time_dim)
        self.down2 = UNetBlock(
            base_channels * 2, base_channels * 4, context_dim + time_dim)

        # Bottleneck
        self.bottleneck = UNetBlock(
            base_channels * 4, base_channels * 4, context_dim + time_dim)

        # Upsampling blocks
        self.up1 = UNetBlock(
            base_channels * 4, base_channels * 2, context_dim + time_dim)
        self.up2 = UNetBlock(
            base_channels * 2, base_channels, context_dim + time_dim)

        # Final convolution
        self.final_conv = nn.Conv2d(
            base_channels, in_channels, kernel_size=1)

    def forward(self, x, text_embedding, time_embedding):
        # Concatenate conditioning signals

        if len(time_embedding.shape) < 2:
            # print(text_embedding.shape, time_embedding.shape)
            time_embedding = torch.stack([time_embedding for _ in range(
                len(text_embedding))], dim=0).to(device=time_embedding.device)
            # print(text_embedding.shape, time_embedding.shape)

        context = torch.cat([text_embedding, time_embedding], dim=-1)

        # Initial convolution
        x = self.init_conv(x)

        # Downsampling path
        x = self.down1(x, context)
        x = F.avg_pool2d(x, 2)
        x = self.down2(x, context)
        x = F.avg_pool2d(x, 2)

        # Bottleneck
        x = self.bottleneck(x, context)

        # Upsampling path
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = self.up1(x, context)
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = self.up2(x, context)

        # Final convolution
        x = self.final_conv(x)
        return x


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
