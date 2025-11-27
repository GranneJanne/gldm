import torch

from diffusion.schedules import BetaSchedule


class LatentDiffusionForward:
    """DDPM-style forward diffusion process in latent space."""

    def __init__(self, schedule: BetaSchedule) -> None:
        self.schedule = schedule

    def q_sample(
        self, z0: torch.Tensor, t: torch.Tensor, eps: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample z_t given z_0 and timestep indices t."""
        if eps is None:
            eps = torch.randn_like(z0)

        device = z0.device
        sqrt_alpha_bar_t = self.schedule.sqrt_alphas_cumprod.to(device)[t].view(-1, 1)
        sqrt_one_minus_alpha_bar_t = self.schedule.sqrt_one_minus_alphas_cumprod.to(
            device
        )[t].view(-1, 1)

        z_t = sqrt_alpha_bar_t * z0 + sqrt_one_minus_alpha_bar_t * eps
        return z_t, eps

    def sample_timesteps(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """
        Uniformly sample integer timesteps t in [0, T-1].
        """
        T = self.schedule.betas.shape[0]
        t = torch.randint(
            low=0, high=T, size=(batch_size,), device=device, dtype=torch.long
        )
        return t
