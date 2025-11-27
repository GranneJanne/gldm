from typing import Dict

import torch


class BetaSchedule:
    """Container for DDPM-style beta schedule and derived quantities."""

    def __init__(self, betas: torch.Tensor) -> None:
        if betas.dim() != 1:
            raise ValueError("betas must be 1D tensor of length T")

        self.device = betas.device
        self.betas = betas
        self.alphas = 1.0 - betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat(
            [torch.ones(1, device=self.device), self.alphas_cumprod[:-1]], dim=0
        )

        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)


def make_beta_schedule(cfg: Dict) -> BetaSchedule:
    """Build a BetaSchedule from a config dict."""
    T = cfg.get("T", 1000)
    schedule_type = cfg.get("schedule", "linear").lower()
    beta_start = cfg.get("beta_start", 1e-4)
    beta_end = cfg.get("beta_end", 2e-2)

    if schedule_type == "linear":
        betas = torch.linspace(beta_start, beta_end, T)
    else:
        raise NotImplementedError(f"Unknown beta schedule: {schedule_type}")

    return BetaSchedule(betas)
