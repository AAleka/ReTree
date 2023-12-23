import torch
from torch import nn
from einops import rearrange
import torch.nn.functional as F


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        x = self.fn(x, **kwargs) + x
        return x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        x = self.norm(x)
        x = self.fn(x, **kwargs)
        return x


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x = self.net(x)
        return x


class LSA(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.temperature = nn.Parameter(torch.log(torch.tensor(dim_head ** -0.5)))

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.temperature.exp()

        mask = torch.eye(dots.shape[-1], device=dots.device, dtype=torch.bool)
        mask_value = -torch.finfo(dots.dtype).max
        dots = dots.masked_fill(mask, mask_value)

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_ratio, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(PreNorm(dim, LSA(dim, heads=heads, dim_head=dim_head, dropout=dropout))),
                Residual(PreNorm(dim, FeedForward(dim, mlp_ratio * dim, dropout=dropout)))
            ]))

    def forward(self, x):
        shape = x.shape

        x = x.flatten(2).transpose(1, 2)

        for attn, ff in self.layers:
            x = attn(x)
            x = ff(x)

        x = x.permute(0, 2, 1).view(shape[0], shape[1], shape[2], shape[3])
        return x


class To_Image(nn.Module):
    def __init__(self, input_channels, output_channels):
        super().__init__()
        self.conv = nn.Conv2d(input_channels, output_channels, kernel_size=1)

    def forward(self, x):
        x = self.conv(x)
        return x


class Patch_Embedding(nn.Module):
    def __init__(self, dim, patch_size, img_channels=3, emb_dim=256):
        super().__init__()
        self.embed = nn.Conv2d(img_channels, dim, kernel_size=patch_size, stride=patch_size)

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_dim,
                dim
            ),
        )

    def forward(self, x, t):
        x = self.embed(x)
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, residual=False):
        super().__init__()
        self.residual = residual
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, mid_channels),
            nn.GELU(),

            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, out_channels),
        )

    def forward(self, x):
        if self.residual:
            return F.gelu(x + self.double_conv(x))
        else:
            return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=256):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(4),
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels),
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_dim,
                out_channels
            ),
        )

    def forward(self, x, t):
        x = self.maxpool_conv(x)
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=256):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = nn.Sequential(
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels, in_channels // 2),
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_dim,
                out_channels
            ),
        )

    def forward(self, x, skip_x=None, t=None, last=False):
        x = self.up(x) if not last else F.interpolate(x, size=x.shape[2]+16, mode="bilinear", align_corners=True)
        if skip_x is not None:
            x = torch.cat([skip_x, x], dim=1)
        x = self.conv(x)
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb


class DDPM(nn.Module):
    def __init__(self, time_dim=256, img_channels=4, dim=2 * 2048, depth=1, heads=4, dim_head=64,
                 mlp_ratio=4, drop_rate=0.):
        super(DDPM, self).__init__()
        self.img_channels = img_channels
        self.dim = dim
        self.time_dim = time_dim
        emb_dim = time_dim
        dims = (dim // 32, dim // 16, dim // 8, dim // 4, dim // 2, dim // 1)

        self.initial = nn.Sequential(
            nn.Conv2d(img_channels, dims[0], kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, dims[0]),
            nn.GELU(),
        )

        self.down_blocks = nn.ModuleList()
        self.up_blocks = nn.ModuleList()
        self.transformer_blocks = nn.ModuleList()
        self.bottleneck_blocks = nn.ModuleList()

        self.down_blocks.append(Down(dims[0], dims[2], emb_dim))
        self.down_blocks.append(Down(dims[2], dims[3], emb_dim))

        self.transformer_blocks.append(Transformer(dims[2], depth, heads, dim_head, mlp_ratio, drop_rate))
        self.transformer_blocks.append(Transformer(dims[3], depth, heads, dim_head, mlp_ratio, drop_rate))

        self.bottleneck_blocks.append(DoubleConv(dims[3], dims[4]))
        self.bottleneck_blocks.append(DoubleConv(dims[4], dims[4]))
        self.bottleneck_blocks.append(DoubleConv(dims[4], dims[3]))

        self.up_blocks.append(Up(dims[3], dims[2], emb_dim))
        self.up_blocks.append(Up(dims[3], dims[1], emb_dim))
        self.up_blocks.append(Up(dims[1], dims[0], emb_dim))
        self.up_blocks.append(Up(dims[1], dims[0], emb_dim))

        self.last = To_Image(dims[0], 3 if img_channels > 1 else 1)

    def pos_encoding(self, t, channels):
        inv_freq = 1.0 / (10000 ** (torch.arange(0, channels, 2).float().to(torch.device("cuda")) / channels))
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc

    def forward(self, x, t, mask=None):
        t = t.unsqueeze(-1).type(torch.float)
        t = self.pos_encoding(t, self.time_dim)

        if self.img_channels == 4:
            x = torch.cat([x, mask], 1)

        x1 = self.initial(x)

        x2 = self.down_blocks[0](x1, t)
        x2 = self.transformer_blocks[0](x2)

        x3 = self.down_blocks[1](x2, t)
        x3 = self.transformer_blocks[1](x3)

        x3 = self.bottleneck_blocks[0](x3)
        x3 = self.bottleneck_blocks[1](x3)
        x3 = self.bottleneck_blocks[2](x3)

        x = self.up_blocks[0](x3, None, t)
        x = self.up_blocks[1](x, x2, t)
        x = self.up_blocks[2](x, None, t)
        x = self.up_blocks[3](x, x1, t)

        x = self.last(x)

        return x


class ConvolutionBlockD(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(ConvolutionBlockD, self).__init__()
        self.convolution = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 4, stride, 1, bias=True, padding_mode="reflect"),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        return self.convolution(x)


class Discriminator(nn.Module):
    def __init__(self, in_channels=3, features=(64, 128, 256, 512), augmentations=None):
        super(Discriminator, self).__init__()
        self.augmentations = augmentations
        self.initial = nn.Sequential(
            nn.Conv2d(in_channels, features[0], 4, stride=2, padding=1, padding_mode="reflect"),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.down_blocks = nn.ModuleList(
            [
                ConvolutionBlockD(features[0], features[1], stride=1),
                Transformer(features[1], 1, 4, 64, 4, 0.),
                ConvolutionBlockD(features[1], features[2], stride=2),
                Transformer(features[2], 1, 4, 64, 4, 0.),
                ConvolutionBlockD(features[2], features[3], stride=2),
                Transformer(features[3], 1, 4, 64, 4, 0.),
            ]
        )

        self.last = nn.Conv2d(features[3], 1, 4, stride=1, padding=1, padding_mode="reflect")

    def forward(self, x):
        if self.augmentations is not None:
            x = self.augmentations(x)

        x = self.initial(x)

        for block in self.down_blocks:
            x = block(x)

        x = self.last(x)

        return torch.sigmoid(x)


if __name__ == '__main__':
    IMAGE_SIZE = 128

    ddpm = DDPM(img_channels=4).to(torch.device("cuda"))
    tensor_x = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE).to(torch.device("cuda"))
    tensor_m = torch.randn(1, 1, IMAGE_SIZE, IMAGE_SIZE).to(torch.device("cuda"))
    tensor_t = tensor_x.new_tensor([500] * tensor_x.shape[0]).long().to(torch.device("cuda"))
    output = ddpm(tensor_x, tensor_t, tensor_m)
    print(output.shape)

    D1 = Discriminator().to(torch.device("cuda"))
    output = D1(tensor_x)
    print(output.shape)
