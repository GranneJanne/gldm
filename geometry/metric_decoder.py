import torch
import torch.nn as nn


def estimate_diag_jtj(
    decoder: nn.Module,
    z: torch.Tensor,
    num_samples: int = 1,
) -> torch.Tensor:
    """Hutchinson estimate of the diagonal of the pullback metric
    G(z) = J(z)^T J(z) for decoder f: Z -> X, using u ~ N(0, I) in image space."""
    z = z.detach().requires_grad_(True)
    B, d = z.shape
    device = z.device

    diag_G = torch.zeros(B, d, device=device)

    for _ in range(num_samples):
        x = decoder(z)
        u = torch.randn_like(x)
        scalar = (x * u).sum()

        grad_z = torch.autograd.grad(
            outputs=scalar,
            inputs=z,
            create_graph=False,
            retain_graph=True,
            only_inputs=True,
        )[0]

        diag_G = diag_G + grad_z.pow(2)

    diag_G = diag_G / float(num_samples)

    return diag_G


class WeightNormalizer(nn.Module):
    """Weights (g / running mean of g) ** power, optionally clipped. Frozen in eval mode."""

    def __init__(self, cfg_geom: dict | None = None) -> None:
        super().__init__()
        cfg_geom = cfg_geom or {}
        self.power = cfg_geom.get("power", 1.0)
        self.clip = (cfg_geom.get("clip_min", None), cfg_geom.get("clip_max", None))
        self.register_buffer("mean", torch.zeros(()))
        self.register_buffer("count", torch.zeros((), dtype=torch.long))

    def forward(self, g: torch.Tensor) -> torch.Tensor:
        if self.training:
            self.count += 1
            self.mean += (g.detach().mean() - self.mean) / self.count
        weights = (g / self.mean).pow(self.power)
        if self.clip != (None, None):
            weights = weights.clamp(*self.clip)
        return weights
