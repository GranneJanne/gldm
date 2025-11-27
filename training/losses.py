import torch
import torch.nn as nn

from geometry.metric_decoder import WeightNormalizer, estimate_diag_jtj


def geometry_weighted_epsilon_loss(
    decoder: nn.Module,
    z0: torch.Tensor,
    eps_pred: torch.Tensor,
    eps_true: torch.Tensor,
    normalizer: WeightNormalizer,
    num_samples: int = 1,
) -> torch.Tensor:
    """GLDM loss: epsilon MSE weighted by the decoder metric at the clean latent z0."""
    g_hat = estimate_diag_jtj(decoder, z0, num_samples=num_samples)
    weights = normalizer(g_hat)  # [B, d]
    return (weights * (eps_pred - eps_true).pow(2)).mean()


def random_weighted_epsilon_loss(
    z0: torch.Tensor,
    eps_pred: torch.Tensor,
    eps_true: torch.Tensor,
    normalizer: WeightNormalizer,
) -> torch.Tensor:
    """Random-weight control: positive i.i.d. weights |N(0,1)|^2 of the same shape as g_hat"""
    weights = normalizer(torch.randn_like(z0).pow(2))  # [B, d]
    return (weights * (eps_pred - eps_true).pow(2)).mean()
