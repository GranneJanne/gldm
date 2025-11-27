import torch
import torch.nn as nn

from diffusion.schedules import BetaSchedule


@torch.no_grad()
def p_sample_ddpm_step(
    eps_model: nn.Module,
    schedule: BetaSchedule,
    z_t: torch.Tensor,
    t_int: int,
) -> torch.Tensor:
    """Single DDPM reverse step in latent space."""
    device = z_t.device
    B, _ = z_t.shape

    t_batch = torch.full((B,), t_int, device=device, dtype=torch.long)

    betas_t = schedule.betas[t_int]  # scalar
    alpha_t = schedule.alphas[t_int]
    alpha_bar_t = schedule.alphas_cumprod[t_int]
    alpha_bar_prev = schedule.alphas_cumprod_prev[t_int]

    sqrt_one_minus_alpha_bar_t = schedule.sqrt_one_minus_alphas_cumprod[t_int]

    eps_pred = eps_model(z_t, t_batch)  # [B, d]

    z0_pred = (z_t - sqrt_one_minus_alpha_bar_t * eps_pred) / torch.sqrt(alpha_bar_t)

    posterior_variance_t = betas_t * (1.0 - alpha_bar_prev) / (1.0 - alpha_bar_t)
    posterior_variance_t = torch.clamp(posterior_variance_t, min=1e-20)

    coef1 = betas_t * torch.sqrt(alpha_bar_prev) / (1.0 - alpha_bar_t)
    coef2 = (1.0 - alpha_bar_prev) * torch.sqrt(alpha_t) / (1.0 - alpha_bar_t)

    posterior_mean_t = coef1 * z0_pred + coef2 * z_t

    if t_int == 0:
        return posterior_mean_t

    noise = torch.randn_like(z_t)
    z_prev = posterior_mean_t + torch.sqrt(posterior_variance_t) * noise
    return z_prev


@torch.no_grad()
def sample_latents_ddpm(
    eps_model: nn.Module,
    schedule: BetaSchedule,
    num_samples: int,
    latent_dim: int,
    device: torch.device,
    z_T: torch.Tensor | None = None,
) -> torch.Tensor:
    """Unconditional DDPM sampling in latent space."""
    eps_model.eval()

    T = schedule.betas.shape[0]

    if z_T is None:
        z_t = torch.randn(num_samples, latent_dim, device=device)
    else:
        z_t = z_T.to(device)

    for t_int in reversed(range(T)):
        z_t = p_sample_ddpm_step(eps_model, schedule, z_t, t_int)

    return z_t
