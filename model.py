import torch
import torch.nn as nn
import torch.nn.functional as F
import math


##########################################################################
# Phase Extraction Module (PEM) — zero learnable parameters
# Copied from Phaseformer: model_with_eca.py, line 13
##########################################################################
def inv_mag(x):
    fft_ = torch.fft.fft2(x)
    fft_ = torch.fft.ifft2(1 * torch.exp(1j * (fft_.angle())))
    return fft_.real


##########################################################################
# Optimized Phase Attention Block (OPAB) for skip connections
# Copied from Phaseformer: model_with_eca.py, line 18
##########################################################################
class ECA(nn.Module):
    def __init__(self, channels, b=1, gamma=2):
        super(ECA, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.channels = channels
        self.b = b
        self.gamma = gamma
        self.conv = nn.Conv1d(
            1, 1,
            kernel_size=self.kernel_size(),
            padding=(self.kernel_size() - 1) // 2,
            bias=False,
        )
        self.sigmoid = nn.Sigmoid()

    def kernel_size(self):
        k = int(abs((math.log2(self.channels) / self.gamma) + self.b / self.gamma))
        out = k if k % 2 else k + 1
        return out

    def forward(self, x):
        x1 = inv_mag(x)
        y = self.avg_pool(x1)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)


##########################################################################
# Phase-based Multi-head Transposed Attention (PMSA)
# Copied from Phaseformer: model_with_eca.py, line 62
##########################################################################
class MDTA(nn.Module):
    def __init__(self, channels, num_heads):
        super(MDTA, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(1, num_heads, 1, 1))

        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1, bias=False)
        self.qkv_conv = nn.Conv2d(
            channels * 3, channels * 3,
            kernel_size=3, padding=1, groups=channels * 3, bias=False,
        )
        self.project_out = nn.Conv2d(channels, channels, kernel_size=1, bias=False)

    def forward(self, x):
        b, c, h, w = x.shape
        q, k, v = self.qkv_conv(self.qkv(x)).chunk(3, dim=1)

        q = inv_mag(q)
        k = inv_mag(k)

        q = q.reshape(b, self.num_heads, -1, h * w)
        k = k.reshape(b, self.num_heads, -1, h * w)
        v = v.reshape(b, self.num_heads, -1, h * w)

        q, k = F.normalize(q, dim=-1), F.normalize(k, dim=-1)

        attn = torch.softmax(
            torch.matmul(q, k.transpose(-2, -1).contiguous()) * self.temperature,
            dim=-1,
        )
        out = self.project_out(torch.matmul(attn, v).reshape(b, -1, h, w))
        return out


##########################################################################
# Gated Depthwise Feed-Forward Network
# Copied from Phaseformer: model_with_eca.py, line 90
##########################################################################
class GDFN(nn.Module):
    def __init__(self, channels, expansion_factor):
        super(GDFN, self).__init__()
        hidden_channels = int(channels * expansion_factor)
        self.project_in = nn.Conv2d(channels, hidden_channels * 2, kernel_size=1, bias=False)
        self.conv = nn.Conv2d(
            hidden_channels * 2, hidden_channels * 2,
            kernel_size=3, padding=1, groups=hidden_channels * 2, bias=False,
        )
        self.project_out = nn.Conv2d(hidden_channels, channels, kernel_size=1, bias=False)

    def forward(self, x):
        x1, x2 = self.conv(self.project_in(x)).chunk(2, dim=1)
        x = self.project_out(F.gelu(x1) * x2)
        return x


##########################################################################
# Phase-based Transformer Block
# Copied from Phaseformer: model_with_eca.py, line 125
##########################################################################
class TransformerBlock(nn.Module):
    def __init__(self, channels, num_heads, expansion_factor):
        super(TransformerBlock, self).__init__()
        self.norm1 = nn.LayerNorm(channels)
        self.attn = MDTA(channels, num_heads)
        self.norm2 = nn.LayerNorm(channels)
        self.ffn = GDFN(channels, expansion_factor)

    def forward(self, x):
        b, c, h, w = x.shape
        x = x + self.attn(
            self.norm1(x.reshape(b, c, -1).transpose(-2, -1).contiguous())
            .transpose(-2, -1).contiguous().reshape(b, c, h, w)
        )
        x = x + self.ffn(
            self.norm2(x.reshape(b, c, -1).transpose(-2, -1).contiguous())
            .transpose(-2, -1).contiguous().reshape(b, c, h, w)
        )
        return x


##########################################################################
# DownSample: Conv3x3(C -> C//2) + PixelUnshuffle(2) => output 2C channels
# Copied from Phaseformer: model_with_eca.py, line 143
##########################################################################
class DownSample(nn.Module):
    def __init__(self, channels):
        super(DownSample, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels // 2, kernel_size=3, padding=1, bias=False),
            nn.PixelUnshuffle(2),
        )

    def forward(self, x):
        return self.body(x)


##########################################################################
# UpSample: Conv3x3(C -> 2C) + PixelShuffle(2) => output C//2 channels
# Copied from Phaseformer: model_with_eca.py, line 153
##########################################################################
class UpSample(nn.Module):
    def __init__(self, channels):
        super(UpSample, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels * 2, kernel_size=3, padding=1, bias=False),
            nn.PixelShuffle(2),
        )

    def forward(self, x):
        return self.body(x)


##########################################################################
# AIDNet-Phase: Lightweight aerial image dehazing network
# 3-level U-Net with phase-based attention, ~2.38M parameters
##########################################################################
class Network(nn.Module):
    def __init__(
        self,
        channels=[48, 96, 192],
        num_blocks=[2, 2, 2, 2, 2],
        num_heads=[1, 2, 4, 2, 1],
        expansion_factor=2.66,
    ):
        super(Network, self).__init__()

        # Input / output projections
        self.input_conv = nn.Conv2d(3, channels[0], kernel_size=3, padding=1, bias=False)
        self.output_conv = nn.Conv2d(channels[0], 3, kernel_size=3, padding=1, bias=False)

        # Encoder level 1
        self.encoder_l1 = nn.Sequential(
            *[TransformerBlock(channels[0], num_heads[0], expansion_factor)
              for _ in range(num_blocks[0])]
        )

        # Encoder level 2
        self.down1 = DownSample(channels[0])
        self.encoder_l2 = nn.Sequential(
            *[TransformerBlock(channels[1], num_heads[1], expansion_factor)
              for _ in range(num_blocks[1])]
        )

        # Bottleneck
        self.down2 = DownSample(channels[1])
        self.bottleneck = nn.Sequential(
            *[TransformerBlock(channels[2], num_heads[2], expansion_factor)
              for _ in range(num_blocks[2])]
        )

        # Decoder level 2
        self.up1 = UpSample(channels[2])
        self.reduce2 = nn.Conv2d(channels[1] * 2, channels[1], kernel_size=1, bias=False)
        self.decoder_l2 = nn.Sequential(
            *[TransformerBlock(channels[1], num_heads[3], expansion_factor)
              for _ in range(num_blocks[3])]
        )

        # Decoder level 1
        self.up2 = UpSample(channels[1])
        self.reduce1 = nn.Conv2d(channels[0] * 2, channels[0], kernel_size=1, bias=False)
        self.decoder_l1 = nn.Sequential(
            *[TransformerBlock(channels[0], num_heads[4], expansion_factor)
              for _ in range(num_blocks[4])]
        )

        # Phase-attention skip connections (ECA / OPAB)
        self.eca_skip_1 = ECA(channels[0])
        self.eca_skip_2 = ECA(channels[1])

    def forward(self, inp):
        x = self.input_conv(inp)

        # Encoder
        enc1 = self.encoder_l1(x)
        skip1 = self.eca_skip_1(enc1)

        enc2 = self.encoder_l2(self.down1(enc1))
        skip2 = self.eca_skip_2(enc2)

        # Bottleneck
        bot = self.bottleneck(self.down2(enc2))

        # Decoder
        dec2 = self.decoder_l2(
            self.reduce2(torch.cat([self.up1(bot), skip2], dim=1))
        )
        dec1 = self.decoder_l1(
            self.reduce1(torch.cat([self.up2(dec2), skip1], dim=1))
        )

        # Residual learning
        return inp + self.output_conv(dec1)
