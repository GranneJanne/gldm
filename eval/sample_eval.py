import os
import torch
from torchvision import utils as vutils

from diffusion.sampling import sample_latents_ddpm
from eval.utils import ensure_dir


@torch.no_grad()
def generate_samples(
    vae,
    eps_model,
    schedule,
    num_samples: int,
    latent_dim: int,
    device: torch.device,
    batch_size: int = 64,
    outdir: str = "",
    prefix: str = "ldm",
    save_individual: bool = False,
) -> dict[str, str]:
    """Generate samples from a trained latent diffusion model."""
    ensure_dir(outdir)
    nrow = int(num_samples**0.5) if int(num_samples**0.5) ** 2 == num_samples else 8

    all_imgs = []

    remaining = num_samples
    while remaining > 0:
        bs = min(batch_size, remaining)
        z0 = sample_latents_ddpm(
            eps_model=eps_model,
            schedule=schedule,
            num_samples=bs,
            latent_dim=latent_dim,
            device=device,
            z_T=None,
        )
        x = vae.decode(z0).cpu()
        all_imgs.append(x)
        remaining -= bs

    all_imgs = torch.cat(all_imgs, dim=0)[:num_samples]

    grid = vutils.make_grid(all_imgs, nrow=nrow, normalize=True, scale_each=True)
    grid_path = os.path.join(outdir, f"samples_{prefix}_{num_samples}.png")
    vutils.save_image(grid, grid_path)

    out = {"grid": grid_path}

    if save_individual:
        imgs_dir = os.path.join(outdir, f"{prefix}_samples")
        ensure_dir(imgs_dir)
        for i in range(num_samples):
            img_path = os.path.join(imgs_dir, f"{prefix}_{i:06d}.png")
            vutils.save_image(all_imgs[i], img_path, normalize=True, scale_each=True)
        out["dir"] = imgs_dir

    return out


@torch.no_grad()
def generate_compare_grid(
    vae,
    eps_model_ldm,
    eps_model_gldm,
    schedule_ldm,
    schedule_gldm,
    num_samples: int,
    latent_dim: int,
    device: torch.device,
    outdir: str,
    nrow: int = 8,
) -> str:
    """Generate a joint comparison grid: first all LDM, then all GLDM samples."""
    ensure_dir(outdir)

    z_T = torch.randn(num_samples, latent_dim, device=device)

    z0_ldm = sample_latents_ddpm(
        eps_model=eps_model_ldm,
        schedule=schedule_ldm,
        num_samples=num_samples,
        latent_dim=latent_dim,
        device=device,
        z_T=z_T,
    )
    z0_gldm = sample_latents_ddpm(
        eps_model=eps_model_gldm,
        schedule=schedule_gldm,
        num_samples=num_samples,
        latent_dim=latent_dim,
        device=device,
        z_T=z_T,
    )

    x_ldm = vae.decode(z0_ldm).cpu()
    x_gldm = vae.decode(z0_gldm).cpu()

    all_imgs = torch.cat([x_ldm, x_gldm], dim=0)

    grid = vutils.make_grid(all_imgs, nrow=nrow, normalize=True, scale_each=True)
    joint_path = os.path.join(outdir, f"samples_COMPARE_LDM_GLDM_{num_samples}.png")
    vutils.save_image(grid, joint_path)
    return joint_path
