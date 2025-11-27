import os

import torch
import yaml

from models.vae import vae_from_config, VAE
from models.latent_unet import LatentUNet
from diffusion.schedules import make_beta_schedule


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def build_vae_from_config(cfg: dict, device: torch.device) -> VAE:
    model_cfg = cfg.get("model", {})
    vae: VAE = vae_from_config(model_cfg).to(device)

    train_cfg = cfg.get("train", {})
    vae_ckpt_path = train_cfg.get("vae_checkpoint", None)
    if vae_ckpt_path is None:
        raise ValueError("Config must contain train.vae_checkpoint for VAE.")

    ckpt = torch.load(vae_ckpt_path, map_location=device)
    state = ckpt.get("state_dict", ckpt)
    vae.load_state_dict(state)
    vae.eval()
    for p in vae.parameters():
        p.requires_grad_(False)
    return vae


def build_eps_model_from_config(
    cfg: dict,
    latent_dim: int,
    device: torch.device,
    ckpt_path: str,
) -> LatentUNet:
    diff_cfg = cfg.get("diffusion", {})
    eps_model = LatentUNet(
        latent_dim=latent_dim,
        base_channels=diff_cfg.get("base_channels", 128),
        channel_mults=diff_cfg.get("channel_mults", [1, 2, 4]),
        time_emb_dim=diff_cfg.get("time_emb_dim", 256),
    ).to(device)

    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt.get("eps_model_state_dict", ckpt)
    eps_model.load_state_dict(state)
    eps_model.eval()
    return eps_model


def build_schedule_from_config(cfg: dict):
    diff_cfg = cfg.get("diffusion", {})
    return make_beta_schedule(diff_cfg)


def get_latent_dim(vae: VAE) -> int:
    if hasattr(vae, "latent_dim"):
        return vae.latent_dim
    raise RuntimeError(
        "VAE must expose .latent_dim for this eval suite or adapt get_latent_dim."
    )


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_metrics_dict(path: str, metrics: dict) -> None:
    import json

    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)
