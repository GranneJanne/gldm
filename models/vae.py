from __future__ import annotations

from typing import List, Optional, Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


class VAE(nn.Module):
    """Convolutional VAE suitable for CelebA-style faces."""

    def __init__(
        self,
        in_channels: int = 3,
        img_size: int = 64,
        latent_dim: int = 128,
        hidden_dims: Optional[List[int]] = None,
        variational: bool = True,
    ) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.img_size = img_size
        self.latent_dim = latent_dim
        self.variational = variational

        if hidden_dims is None:
            hidden_dims = [64, 128, 256, 512]

        self.hidden_dims = hidden_dims

        encoder_modules: List[nn.Module] = []
        current_channels = in_channels

        for h_dim in hidden_dims:
            encoder_modules.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels=current_channels,
                        out_channels=h_dim,
                        kernel_size=4,
                        stride=2,
                        padding=1,
                    ),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU(0.2, inplace=True),
                )
            )
            current_channels = h_dim

        self.encoder = nn.Sequential(*encoder_modules)

        factor = 2 ** len(hidden_dims)
        if img_size % factor != 0:
            raise ValueError(
                f"img_size={img_size} is not divisible by 2^{len(hidden_dims)}={factor}. "
                "Choose img_size or hidden_dims so the final spatial size is integer "
                "(e.g. img_size=64 with 4 downsampling layers)."
            )

        self.final_height = img_size // factor
        self.final_width = img_size // factor
        encoder_output_dim = hidden_dims[-1] * self.final_height * self.final_width

        self.fc_mu = nn.Linear(encoder_output_dim, latent_dim)
        self.fc_logvar = nn.Linear(encoder_output_dim, latent_dim)

        self.decoder_input = nn.Linear(latent_dim, encoder_output_dim)

        decoder_modules: List[nn.Module] = []
        hidden_dims_rev = hidden_dims[::-1]

        for i in range(len(hidden_dims_rev) - 1):
            decoder_modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(
                        in_channels=hidden_dims_rev[i],
                        out_channels=hidden_dims_rev[i + 1],
                        kernel_size=4,
                        stride=2,
                        padding=1,
                        output_padding=0,
                    ),
                    nn.BatchNorm2d(hidden_dims_rev[i + 1]),
                    nn.LeakyReLU(0.2, inplace=True),
                )
            )

        self.decoder = nn.Sequential(*decoder_modules)

        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=hidden_dims_rev[-1],
                out_channels=in_channels,
                kernel_size=4,
                stride=2,
                padding=1,
                output_padding=0,
            ),
            nn.Tanh(),
        )

    def encode(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode input image batch x into latent parameters and sample."""
        enc = self.encoder(x)  # [B, hidden_dims[-1], H_final, W_final]
        enc = torch.flatten(enc, start_dim=1)  # [B, D_enc]

        mu = self.fc_mu(enc)
        if self.variational:
            logvar = self.fc_logvar(enc)
            z = self.reparameterize(mu, logvar)
        else:
            logvar = torch.zeros_like(mu)
            z = mu

        return mu, logvar, z

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick to sample z ~ N(mu, sigma^2 I)."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent z to reconstruction."""
        dec_input = self.decoder_input(z)  # [B, D_enc]
        dec_input = dec_input.view(
            -1, self.hidden_dims[-1], self.final_height, self.final_width
        )  # [B, hidden_dims[-1], H_final, W_final]

        x = self.decoder(dec_input)
        x = self.final_layer(x)  # [B, C, H, W], in [-1, 1]
        return x

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        mu, logvar, z = self.encode(x)
        recon = self.decode(z)
        return {"recon": recon, "mu": mu, "logvar": logvar, "z": z}

    @staticmethod
    def loss_function(
        recon_x: torch.Tensor,
        x: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        kld_weight: float = 1.0,
        recon_loss_type: str = "l1",
    ) -> Dict[str, torch.Tensor]:
        """Compute VAE loss = reconstruction_term + KLD_term."""
        if recon_loss_type.lower() == "mse":
            recon_loss = F.mse_loss(recon_x, x, reduction="mean")
        elif recon_loss_type.lower() == "l1":
            recon_loss = F.l1_loss(recon_x, x, reduction="mean")
        else:
            raise ValueError(f"Unknown recon_loss_type: {recon_loss_type}")

        kld_loss = -0.5 * torch.mean(
            torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
        )

        loss = recon_loss + kld_weight * kld_loss
        return {
            "loss": loss,
            "recon_loss": recon_loss,
            "kld_loss": kld_loss,
        }

    def sample(
        self, num_samples: int, device: Optional[torch.device] = None
    ) -> torch.Tensor:
        """Sample new images from the prior p(z) = N(0, I) and decode."""
        if device is None:
            device = next(self.parameters()).device

        z = torch.randn(num_samples, self.latent_dim, device=device)
        samples = self.decode(z)
        return samples

    def generate(self, x: torch.Tensor) -> torch.Tensor:
        """Reconstruct x by encoding and decoding."""
        out = self.forward(x)
        return out["recon"]


def vae_from_config(cfg: Dict) -> VAE:
    """Convenience constructor: build VAE from a config dict."""
    return VAE(
        in_channels=cfg.get("in_channels", 3),
        img_size=cfg.get("img_size", 64),
        latent_dim=cfg.get("latent_dim", 128),
        hidden_dims=cfg.get("hidden_dims", None),
        variational=cfg.get("variational", True),
    )
