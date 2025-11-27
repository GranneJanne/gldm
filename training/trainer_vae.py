import os
from typing import Dict, Any

import torch
from torch import optim
from torchvision import utils as vutils
from tqdm import tqdm

from models.vae import VAE, vae_from_config
from dataloaders import make_cifar10_splits, make_celeba_splits


class VAETrainer:
    """Basic VAE trainer."""

    def __init__(self, cfg: Dict[str, Any]) -> None:
        self.cfg = cfg

        self.model_cfg = cfg.get("model", {})
        self.train_cfg = cfg.get("train", {})
        self.data_cfg = cfg.get("data", {})

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.vae: VAE = vae_from_config(self.model_cfg).to(self.device)

        self.optimizer = optim.Adam(
            self.vae.parameters(),
            lr=self.train_cfg.get("lr", 2e-4),
            betas=(0.5, 0.999),
        )

        self.train_loader, self.val_loader, self.test_loader = self._build_dataloaders()

        self.checkpoint_dir = self.train_cfg.get("checkpoint_dir", "checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        self.eval_dir = self.train_cfg.get("eval_dir", "eval_samples")
        os.makedirs(self.eval_dir, exist_ok=True)

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
        num_epochs = self.train_cfg.get("epochs", 20)
        kld_weight = self.train_cfg.get("kld_weight", 1.0)
        recon_loss_type = self.train_cfg.get("recon_loss_type", "mse")

        step = 0

        for epoch in range(1, num_epochs + 1):
            self.vae.train()
            pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}/{num_epochs}")

            for x, _ in pbar:
                x = x.to(self.device)

                out = self.vae(x)
                losses = VAE.loss_function(
                    recon_x=out["recon"],
                    x=x,
                    mu=out["mu"],
                    logvar=out["logvar"],
                    kld_weight=kld_weight,
                    recon_loss_type=recon_loss_type,
                )

                loss = losses["loss"]

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                step += 1
                pbar.set_postfix(
                    loss=f"{loss.item():.4f}",
                    recon=f"{losses['recon_loss'].item():.4f}",
                    kld=f"{losses['kld_loss'].item():.4f}",
                )

            eval_metrics = self.evaluate(epoch)
            print(f"[Eval @ epoch {epoch}] {eval_metrics}")

            ckpt_path = os.path.join(self.checkpoint_dir, f"vae_epoch{epoch}.pt")
            torch.save(
                {
                    "epoch": epoch,
                    "state_dict": self.vae.state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                    "config": self.cfg,
                    "eval_metrics": eval_metrics,
                },
                ckpt_path,
            )
            print(f"Saved VAE checkpoint: {ckpt_path}")

    @torch.no_grad()
    def evaluate(self, epoch: int) -> Dict[str, float]:
        self.vae.eval()

        kld_weight = self.train_cfg.get("kld_weight", 1.0)
        recon_loss_type = self.train_cfg.get("recon_loss_type", "mse")

        val_loss = 0.0
        val_recon = 0.0
        val_kld = 0.0
        count = 0

        last_x = None
        last_out = None

        for x, _ in self.val_loader:
            x = x.to(self.device)
            out = self.vae(x)

            losses = VAE.loss_function(
                recon_x=out["recon"],
                x=x,
                mu=out["mu"],
                logvar=out["logvar"],
                kld_weight=kld_weight,
                recon_loss_type=recon_loss_type,
            )

            val_loss += losses["loss"].item()
            val_recon += losses["recon_loss"].item()
            val_kld += losses["kld_loss"].item()
            count += 1

            last_x = x
            last_out = out

        val_loss /= count
        val_recon /= count
        val_kld /= count

        if last_x is not None and last_out is not None:
            x_real = last_x[:8].cpu()
            x_recon = last_out["recon"][:8].cpu()

            grid = vutils.make_grid(
                torch.cat([x_real, x_recon], dim=0),
                nrow=8,
                normalize=True,
                scale_each=True,
            )
            grid_path = os.path.join(self.eval_dir, f"recon_epoch{epoch}.png")
            vutils.save_image(grid, grid_path)
            print(f"Saved recon grid: {grid_path}")

        return {
            "val_loss": val_loss,
            "val_recon": val_recon,
            "val_kld": val_kld,
        }
