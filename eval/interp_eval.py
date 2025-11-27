import os
import torch
from torchvision import utils as vutils

from diffusion.sampling import sample_latents_ddpm
from eval.lpips_psnr import compute_pairwise_metrics
from eval.utils import ensure_dir


@torch.no_grad()
def latent_interpolation_grid(
    vae,
    eps_model,
    schedule,
    latent_dim: int,
    device: torch.device,
    steps: int,
    out_path: str,
    zA: torch.Tensor | None = None,
    zB: torch.Tensor | None = None,
) -> dict[str, float]:
    """
    Create a latent interpolation between zA and zB, decode to images,
    save a grid, and compute average pairwise metrics (LPIPS/L2/PSNR)
    between consecutive frames.
    """
    assert steps >= 2, "Need at least two steps for interpolation."

    ensure_dir(os.path.dirname(out_path))

    if zA is None:
        zA = sample_latents_ddpm(
            eps_model=eps_model,
            schedule=schedule,
            num_samples=1,
            latent_dim=latent_dim,
            device=device,
            z_T=None,
        )  # [1, d]
    if zB is None:
        zB = sample_latents_ddpm(
            eps_model=eps_model,
            schedule=schedule,
            num_samples=1,
            latent_dim=latent_dim,
            device=device,
            z_T=None,
        )  # [1, d]

    alphas = torch.linspace(0.0, 1.0, steps, device=device)
    zs = [(1.0 - a) * zA + a * zB for a in alphas]
    z_path = torch.cat(zs, dim=0)  # [steps, d]

    x_path = vae.decode(z_path)  # [steps, C, H, W]
    x_path = x_path.detach().cpu()

    grid = vutils.make_grid(
        x_path,
        nrow=steps,
        normalize=True,
        scale_each=True,
    )
    vutils.save_image(grid, out_path)

    x0 = x_path[0:1]
    x1 = x_path[1:2]
    probe_metrics = compute_pairwise_metrics(x0, x1)

    diffs_metrics: dict[str, list[float]] = {}
    for k, v in probe_metrics.items():
        if v is not None:
            diffs_metrics[k] = []

    for i in range(steps - 1):
        a = x_path[i : i + 1]
        b = x_path[i + 1 : i + 2]
        m = compute_pairwise_metrics(a, b)
        for k in diffs_metrics.keys():
            v = m.get(k, None)
            if v is None:
                continue
            val = float(v)
            if torch.isnan(torch.tensor(val)) or torch.isinf(torch.tensor(val)):
                continue
            diffs_metrics[k].append(val)

    summary: dict[str, float] = {}
    for k, vals in diffs_metrics.items():
        if len(vals) == 0:
            summary[k] = float("nan")
        else:
            summary[k] = float(sum(vals) / len(vals))

    return summary
