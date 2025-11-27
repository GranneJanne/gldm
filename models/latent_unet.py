from __future__ import annotations

from typing import List

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def sinusoidal_time_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    """Standard sinusoidal time embedding used in many diffusion models."""
    half_dim = dim // 2
    emb = math.log(10000.0) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, device=timesteps.device) * -emb)
    emb = timesteps.float().unsqueeze(1) * emb.unsqueeze(0)
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb  # [B, dim]


class TimeEmbedMLP(nn.Module):
    """Small MLP to project sinusoidal time embeddings."""

    def __init__(self, time_dim: int, out_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(time_dim, out_dim),
            nn.SiLU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, t_emb: torch.Tensor) -> torch.Tensor:
        return self.net(t_emb)


class ResBlock1D(nn.Module):
    """1D residual block with FiLM conditioning on time embedding."""

    def __init__(self, in_channels: int, out_channels: int, time_emb_dim: int) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.norm1 = nn.GroupNorm(8, in_channels)
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)

        self.time_mlp = nn.Linear(
            time_emb_dim, out_channels * 2
        )  # gamma, beta for FiLM

        if in_channels != out_channels:
            self.shortcut = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        # x: [B, C, L]; t_emb: [B, time_emb_dim]
        h = self.norm1(x)
        h = F.silu(h)
        h = self.conv1(h)
        # FiLM conditioning
        t_out = self.time_mlp(t_emb)  # [B, 2*out_channels]
        gamma, beta = torch.chunk(t_out, 2, dim=1)  # [B, C], [B, C]
        gamma = gamma.unsqueeze(-1)  # [B, C, 1]
        beta = beta.unsqueeze(-1)  # [B, C, 1]

        h = self.norm2(h)
        h = gamma * h + beta
        h = F.silu(h)
        h = self.conv2(h)

        return h + self.shortcut(x)


class Downsample1D(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv1d(channels, channels, kernel_size=4, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample1D(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.ConvTranspose1d(
            channels, channels, kernel_size=4, stride=2, padding=1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class LatentUNet(nn.Module):
    """1D U-Net operating on latent vectors z in R^d."""

    def __init__(
        self,
        latent_dim: int,
        base_channels: int = 64,
        channel_mults: List[int] | None = None,
        time_emb_dim: int = 256,
    ) -> None:
        super().__init__()

        self.latent_dim = latent_dim

        if channel_mults is None:
            channel_mults = [1, 2, 4]

        self.time_emb_dim = time_emb_dim
        self.time_mlp = TimeEmbedMLP(time_dim=time_emb_dim, out_dim=time_emb_dim)

        self.init_conv = nn.Conv1d(1, base_channels, kernel_size=1)

        in_ch = base_channels
        downs = []
        self.down_channels = []

        for mult in channel_mults:
            out_ch = base_channels * mult
            downs.append(ResBlock1D(in_ch, out_ch, time_emb_dim=time_emb_dim))
            self.down_channels.append(out_ch)
            in_ch = out_ch
            if mult != channel_mults[-1]:
                downs.append(Downsample1D(in_ch))

        self.down = nn.ModuleList(downs)

        self.mid_block1 = ResBlock1D(in_ch, in_ch, time_emb_dim=time_emb_dim)
        self.mid_block2 = ResBlock1D(in_ch, in_ch, time_emb_dim=time_emb_dim)

        ups = []
        up_channels = list(reversed(self.down_channels))
        in_ch = up_channels[0]

        for mult in channel_mults[::-1]:
            out_ch = base_channels * mult
            ups.append(ResBlock1D(in_ch + out_ch, out_ch, time_emb_dim=time_emb_dim))
            in_ch = out_ch
            if mult != channel_mults[0]:
                ups.append(Upsample1D(in_ch))

        self.up = nn.ModuleList(ups)

        self.out_norm = nn.GroupNorm(8, in_ch)
        self.out_conv = nn.Conv1d(in_ch, 1, kernel_size=1)

    def forward(self, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        _, L = z.shape
        if L != self.latent_dim:
            raise ValueError(f"Expected latent_dim={self.latent_dim}, got {L}")

        # Time embedding
        t_sin = sinusoidal_time_embedding(t, self.time_emb_dim)  # [B, time_emb_dim]
        t_emb = self.time_mlp(t_sin)  # [B, time_emb_dim]

        x = z.unsqueeze(1)  # [B, 1, L]
        x = self.init_conv(x)  # [B, C, L]

        hs = []
        h = x
        for layer in self.down:
            if isinstance(layer, ResBlock1D):
                h = layer(h, t_emb)
                hs.append(h)
            else:
                h = layer(h)

        h = self.mid_block1(h, t_emb)
        h = self.mid_block2(h, t_emb)

        for layer in self.up:
            if isinstance(layer, ResBlock1D):
                skip = hs.pop()
                h = torch.cat([h, skip], dim=1)
                h = layer(h, t_emb)
            else:
                h = layer(h)

        h = self.out_norm(h)
        h = F.silu(h)
        h = self.out_conv(h)  # [B, 1, L]
        eps_pred = h.squeeze(1)  # [B, L]
        return eps_pred
