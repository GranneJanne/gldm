import os
from typing import Dict

import torch
from torchvision import utils as vutils

from models.vae import vae_from_config, VAE
from models.latent_unet import LatentUNet
from diffusion.schedules import make_beta_schedule
from diffusion.sampling import sample_latents_ddpm


@torch.no_grad()
def run_compare_samples(
    ldm_cfg: Dict,
    ldm_ckpt: str,
    gldm_cfg: Dict,
    gldm_ckpt: str,
    num_samples: int,
    nrow: int,
    outdir: str,
) -> Dict[str, str]:
    """
    Generate baseline LDM and GLDM samples using the same initial latent noise
    and the same VAE decoder. Saves three grids:

      - ldm:   only LDM samples
      - gldm:  only GLDM samples
      - joint: LDM on top, GLDM below (for direct visual comparison)

    Returns:
        dict with keys: 'ldm', 'gldm', 'joint' mapping to image file paths.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_cfg = ldm_cfg.get("model", {})
    diff_cfg_ldm = ldm_cfg.get("diffusion", {})
    diff_cfg_gldm = gldm_cfg.get("diffusion", {})
    train_cfg_ldm = ldm_cfg.get("train", {})
    train_cfg_gldm = gldm_cfg.get("train", {})

    vae: VAE = vae_from_config(model_cfg).to(device)

    vae_ckpt_ldm = train_cfg_ldm.get("vae_checkpoint", None)
    vae_ckpt_gldm = train_cfg_gldm.get("vae_checkpoint", None)
    if vae_ckpt_ldm is None or vae_ckpt_gldm is None:
        raise ValueError("Both LDM and GLDM configs must define train.vae_checkpoint")

    if vae_ckpt_ldm != vae_ckpt_gldm:
        print(
            "[Warning] LDM and GLDM use different VAE checkpoints. "
            "Using LDM's VAE checkpoint for comparison."
        )
    vae_ckpt_path = vae_ckpt_ldm

    vae_ckpt = torch.load(vae_ckpt_path, map_location=device)
    vae_state = vae_ckpt["state_dict"] if "state_dict" in vae_ckpt else vae_ckpt
    vae.load_state_dict(vae_state)
    vae.eval()

    latent_dim = vae.latent_dim

    eps_model_ldm = LatentUNet(
        latent_dim=latent_dim,
        base_channels=diff_cfg_ldm.get("base_channels", 128),
        channel_mults=diff_cfg_ldm.get("channel_mults", [1, 2, 4]),
        time_emb_dim=diff_cfg_ldm.get("time_emb_dim", 256),
    ).to(device)
    ckpt_ldm = torch.load(ldm_ckpt, map_location=device)
    eps_model_ldm.load_state_dict(ckpt_ldm["eps_model_state_dict"])
    eps_model_ldm.eval()

    eps_model_gldm = LatentUNet(
        latent_dim=latent_dim,
        base_channels=diff_cfg_gldm.get("base_channels", 128),
        channel_mults=diff_cfg_gldm.get("channel_mults", [1, 2, 4]),
        time_emb_dim=diff_cfg_gldm.get("time_emb_dim", 256),
    ).to(device)
    ckpt_gldm = torch.load(gldm_ckpt, map_location=device)
    eps_model_gldm.load_state_dict(ckpt_gldm["eps_model_state_dict"])
    eps_model_gldm.eval()

    schedule_ldm = make_beta_schedule(diff_cfg_ldm)
    schedule_gldm = make_beta_schedule(diff_cfg_gldm)

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

    os.makedirs(outdir, exist_ok=True)

    ldm_path = os.path.join(outdir, f"samples_LDM_{num_samples}.png")
    gldm_path = os.path.join(outdir, f"samples_GLDM_{num_samples}.png")

    grid_ldm = vutils.make_grid(x_ldm, nrow=nrow, normalize=True, scale_each=True)
    grid_gldm = vutils.make_grid(x_gldm, nrow=nrow, normalize=True, scale_each=True)
    vutils.save_image(grid_ldm, ldm_path)
    vutils.save_image(grid_gldm, gldm_path)

    all_imgs = torch.cat([x_ldm, x_gldm], dim=0)
    joint_path = os.path.join(outdir, f"samples_COMPARE_LDM_GLDM_{num_samples}.png")
    grid_joint = vutils.make_grid(all_imgs, nrow=nrow, normalize=True, scale_each=True)
    vutils.save_image(grid_joint, joint_path)

    return {"ldm": ldm_path, "gldm": gldm_path, "joint": joint_path}
