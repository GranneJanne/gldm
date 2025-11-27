import os
from typing import Dict

import torch
import torch.nn.functional as F

from models.vae import vae_from_config, VAE
from models.latent_unet import LatentUNet
from diffusion.schedules import make_beta_schedule
from diffusion.forward import LatentDiffusionForward
from dataloaders import make_cifar10_splits


@torch.no_grad()
def _eval_eps_mse_for_model(
    cfg: Dict,
    ckpt_path: str,
    vae: VAE,
    device: torch.device,
) -> float:
    """
    Compute epsilon MSE on CIFAR-10 test set for a single latent diffusion model,
    using a shared VAE and Euclidean epsilon loss.

    Args:
        cfg:       config dict ('ldm' or 'gldm').
        ckpt_path: path to checkpoint.
        vae:       pretrained VAE (already on device, eval mode).
        device:    torch device.

    Returns:
        avg epsilon MSE over test set.
    """
    model_cfg = cfg.get("model", {})
    diff_cfg = cfg.get("diffusion", {})
    train_cfg = cfg.get("train", {})
    data_cfg = cfg.get("data", {})
    seed = cfg.get("seed", 42)

    latent_dim = vae.latent_dim

    eps_model = LatentUNet(
        latent_dim=latent_dim,
        base_channels=diff_cfg.get("base_channels", 128),
        channel_mults=diff_cfg.get("channel_mults", [1, 2, 4]),
        time_emb_dim=diff_cfg.get("time_emb_dim", 256),
    ).to(device)

    ckpt = torch.load(ckpt_path, map_location=device)
    eps_state = ckpt["eps_model_state_dict"]
    eps_model.load_state_dict(eps_state)
    eps_model.eval()

    schedule = make_beta_schedule(diff_cfg)
    forward_process = LatentDiffusionForward(schedule)

    img_size = model_cfg.get("img_size", 32)
    batch_size = train_cfg.get("batch_size", 128)
    num_workers = data_cfg.get("num_workers", 4)
    data_root = data_cfg.get("root", "./data")
    val_fraction = data_cfg.get("val_fraction", 0.1)

    _, _, test_loader = make_cifar10_splits(
        data_root=data_root,
        img_size=img_size,
        batch_size=batch_size,
        num_workers=num_workers,
        val_fraction=val_fraction,
        seed=seed,
    )

    total_loss = 0.0
    count = 0

    for x, _ in test_loader:
        x = x.to(device)

        out_vae = vae(x)
        z0 = out_vae["z"]  # [B, latent_dim]
        B = z0.shape[0]

        t = forward_process.sample_timesteps(B, device)
        z_t, eps_true = forward_process.q_sample(z0, t)

        eps_pred = eps_model(z_t, t)

        loss = F.mse_loss(eps_pred, eps_true)
        total_loss += loss.item()
        count += 1

    avg_loss = total_loss / max(count, 1)
    return avg_loss


@torch.no_grad()
def run_compare_eps_mse(
    ldm_cfg: Dict,
    ldm_ckpt: str,
    gldm_cfg: Dict,
    gldm_ckpt: str,
    outdir: str,
) -> Dict[str, float]:
    """
    Compare baseline LDM vs GLDM on epsilon MSE over CIFAR-10 test set.

    Returns:
        dict with keys:
          - 'eps_mse_ldm'
          - 'eps_mse_gldm'
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_cfg = ldm_cfg.get("model", {})
    train_cfg_ldm = ldm_cfg.get("train", {})
    train_cfg_gldm = gldm_cfg.get("train", {})

    vae_ckpt_ldm = train_cfg_ldm.get("vae_checkpoint", None)
    vae_ckpt_gldm = train_cfg_gldm.get("vae_checkpoint", None)

    if vae_ckpt_ldm is None or vae_ckpt_gldm is None:
        raise ValueError("Both LDM and GLDM configs must define train.vae_checkpoint")
    if vae_ckpt_ldm != vae_ckpt_gldm:
        print(
            "[Warning] LDM and GLDM use different VAE checkpoints. "
            "Using LDM's VAE checkpoint for comparison metrics."
        )

    vae_ckpt_path = vae_ckpt_ldm

    vae: VAE = vae_from_config(model_cfg).to(device)
    vae_ckpt = torch.load(vae_ckpt_path, map_location=device)
    vae_state = vae_ckpt["state_dict"] if "state_dict" in vae_ckpt else vae_ckpt
    vae.load_state_dict(vae_state)
    vae.eval()

    os.makedirs(outdir, exist_ok=True)

    eps_mse_ldm = _eval_eps_mse_for_model(ldm_cfg, ldm_ckpt, vae, device)
    eps_mse_gldm = _eval_eps_mse_for_model(gldm_cfg, gldm_ckpt, vae, device)

    metrics = {
        "eps_mse_ldm": eps_mse_ldm,
        "eps_mse_gldm": eps_mse_gldm,
    }

    metrics_path = os.path.join(outdir, "eps_mse_compare.txt")
    with open(metrics_path, "w") as f:
        for k, v in metrics.items():
            f.write(f"{k}: {v:.6f}\n")

    return metrics
