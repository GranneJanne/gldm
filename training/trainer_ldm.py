import os
from typing import Dict, Any

import torch
import torch.nn.functional as F
from torch import optim
from tqdm import tqdm

from models.vae import VAE, vae_from_config
from models.latent_unet import LatentUNet
from diffusion.schedules import make_beta_schedule
from diffusion.forward import LatentDiffusionForward
from geometry.metric_decoder import WeightNormalizer
from training.losses import geometry_weighted_epsilon_loss, random_weighted_epsilon_loss
from dataloaders import make_cifar10_splits, make_celeba_splits


class LatentDiffusionTrainer:
    """Unified trainer for latent diffusion."""

    def __init__(self, cfg: Dict[str, Any]) -> None:
        self.cfg = cfg

        self.run_type = self.cfg.get("type", "").lower()
        if self.run_type not in ("ldm", "gldm", "random"):
            raise ValueError(
                f"LatentDiffusionTrainer expects cfg['type'] in {{'ldm','gldm','random'}}, "
                f"got: {self.run_type}"
            )

        self.model_cfg = cfg.get("model", {})
        self.diff_cfg = cfg.get("diffusion", {})
        self.train_cfg = cfg.get("train", {})
        self.data_cfg = cfg.get("data", {})
        self.geom_cfg = cfg.get("geometry") or {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.vae: VAE = vae_from_config(self.model_cfg).to(self.device)
        vae_ckpt = self.train_cfg.get("vae_checkpoint", None)
        if vae_ckpt is None or not os.path.isfile(vae_ckpt):
            raise ValueError(
                "You must provide a valid 'vae_checkpoint' in train.vae_checkpoint "
                f"(got: {vae_ckpt})"
            )

        ckpt = torch.load(vae_ckpt, map_location=self.device)
        state_dict = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
        self.vae.load_state_dict(state_dict)
        self.vae.eval()
        for p in self.vae.parameters():
            p.requires_grad_(False)

        latent_dim = self.vae.latent_dim

        self.eps_model = LatentUNet(
            latent_dim=latent_dim,
            base_channels=self.diff_cfg.get("base_channels", 128),
            channel_mults=self.diff_cfg.get("channel_mults", [1, 2, 4]),
            time_emb_dim=self.diff_cfg.get("time_emb_dim", 256),
        ).to(self.device)

        # Loss weight normaliser for gldm and random (unused by ldm).
        self.normalizer = WeightNormalizer(self.geom_cfg).to(self.device)

        beta_schedule = make_beta_schedule(self.diff_cfg)
        self.forward_process = LatentDiffusionForward(beta_schedule)

        self.optimizer = optim.Adam(
            self.eps_model.parameters(),
            lr=self.train_cfg.get("lr", 2e-4),
            betas=(0.9, 0.999),
        )

        self.train_loader, self.val_loader, self.test_loader = self._build_dataloaders()

        default_ckpt_dir = f"experiments/checkpoints/{self.run_type}"
        self.checkpoint_dir = self.train_cfg.get("checkpoint_dir", default_ckpt_dir)
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def _loss(self, z0, eps_pred, eps_true) -> torch.Tensor:
        if self.run_type == "gldm":
            return geometry_weighted_epsilon_loss(
                decoder=self.vae.decode,
                z0=z0,
                eps_pred=eps_pred,
                eps_true=eps_true,
                normalizer=self.normalizer,
                num_samples=self.geom_cfg.get("num_samples", 1),
            )
        if self.run_type == "random":
            return random_weighted_epsilon_loss(
                z0=z0,
                eps_pred=eps_pred,
                eps_true=eps_true,
                normalizer=self.normalizer,
            )
        return F.mse_loss(eps_pred, eps_true)

    def _build_dataloaders(self):
        dataset = self.data_cfg.get("dataset", "cifar10").lower()
        img_size = self.model_cfg.get("img_size", 32)
        batch_size = self.train_cfg.get("batch_size", 128)
        num_workers = self.data_cfg.get("num_workers", 4)
        data_root = self.data_cfg.get("root", "./data")
        val_fraction = self.data_cfg.get("val_fraction", 0.1)
        seed = self.cfg.get("seed", 42)

        if dataset == "cifar10":
            return make_cifar10_splits(
                data_root=data_root,
                img_size=img_size,
                batch_size=batch_size,
                num_workers=num_workers,
                val_fraction=val_fraction,
                seed=seed,
            )
        elif dataset == "celeba":
            return make_celeba_splits(
                data_root=data_root,
                img_size=img_size,
                batch_size=batch_size,
                num_workers=num_workers,
                val_fraction=val_fraction,
                seed=seed,
            )
        else:
            raise ValueError(f"Unknown dataset '{dataset}'")

    def train(self) -> None:
        num_epochs = self.train_cfg.get("epochs", 50)
        log_interval = self.train_cfg.get("log_interval", 100)

        global_step = 0

        for epoch in range(1, num_epochs + 1):
            self.eps_model.train()
            self.normalizer.train()
            pbar = tqdm(
                self.train_loader,
                desc=f"{self.run_type.upper()} Epoch {epoch}/{num_epochs}",
            )

            for batch_idx, (x, _) in enumerate(pbar):
                x = x.to(self.device)

                with torch.no_grad():
                    out_vae = self.vae(x)
                    z0 = out_vae["z"]  # [B, latent_dim]

                B = z0.shape[0]

                t = self.forward_process.sample_timesteps(B, self.device)  # [B]
                z_t, eps_true = self.forward_process.q_sample(z0, t)

                eps_pred = self.eps_model(z_t, t)
                loss = self._loss(z0, eps_pred, eps_true)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                global_step += 1
                if batch_idx % log_interval == 0:
                    pbar.set_postfix(loss=f"{loss.item():.4f}")

            eval_metrics = self.evaluate()
            print(f"[{self.run_type.upper()} Eval @ epoch {epoch}] {eval_metrics}")

            ckpt_path = os.path.join(
                self.checkpoint_dir, f"{self.run_type}_epoch{epoch}.pt"
            )
            torch.save(
                {
                    "epoch": epoch,
                    "step": global_step,
                    "eps_model_state_dict": self.eps_model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "config": self.cfg,
                    "eval_metrics": eval_metrics,
                },
                ckpt_path,
            )
            print(f"Saved {self.run_type.upper()} checkpoint: {ckpt_path}")

    def evaluate(self) -> Dict[str, float]:
        """Evaluate on the validation set."""
        self.eps_model.eval()
        self.normalizer.eval()
        self.vae.eval()

        total_loss = 0.0
        count = 0

        for x, _ in self.val_loader:
            x = x.to(self.device)

            out_vae = self.vae(x)
            z0 = out_vae["z"]

            B = z0.shape[0]
            t = self.forward_process.sample_timesteps(B, self.device)
            z_t, eps_true = self.forward_process.q_sample(z0, t)

            eps_pred = self.eps_model(z_t, t)
            loss = self._loss(z0, eps_pred, eps_true)

            total_loss += loss.item()
            count += 1

        avg_loss = total_loss / max(count, 1)
        key = {
            "ldm": "val_mse_loss",
            "gldm": "val_geom_loss",
            "random": "val_random_loss",
        }[self.run_type]
        return {key: avg_loss}
