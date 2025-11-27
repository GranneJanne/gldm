import os
import numpy as np
import torch
from torch import nn
from torchvision import models, transforms
from PIL import Image

from eval.utils import ensure_dir, save_metrics_dict


def debug_stats(t: torch.Tensor, label: str):
    """Print min/max/mean of a batch for debugging FID issues."""
    print(
        f"[DEBUG] {label}: min={t.min().item():.4f}, "
        f"max={t.max().item():.4f}, mean={t.mean().item():.4f}"
    )


class InceptionFeatureExtractor(nn.Module):
    """
    Loads images from folder, converts them to [0,1], resizes to 299x299,
    and feeds them through InceptionV3 to extract the pool_3 activations.
    Includes debugging prints to diagnose normalization issues.
    """

    def __init__(self, device: torch.device):
        super().__init__()
        inception = models.inception_v3(pretrained=True, transform_input=False)
        inception.fc = nn.Identity()
        inception.eval()
        for p in inception.parameters():
            p.requires_grad_(False)

        self.inception = inception.to(device)
        self.device = device

        self.transform = transforms.Compose(
            [
                transforms.Resize((299, 299)),
                transforms.ToTensor(),  # -> [0,1]
            ]
        )

    @torch.no_grad()
    def get_activations(self, img_paths, batch_size: int = 32, debug=False):
        activations = []
        first_batch = True

        for i in range(0, len(img_paths), batch_size):
            batch_paths = img_paths[i : i + batch_size]
            imgs = []

            for p in batch_paths:
                img = Image.open(p).convert("RGB")
                img = self.transform(img)
                imgs.append(img)

            x = torch.stack(imgs, dim=0).to(self.device)

            if debug and first_batch:
                debug_stats(
                    x, "FIRST REAL/FAKE BATCH BEFORE INCEPTION (expected [0,1])"
                )
                first_batch = False

            act = self.inception(x)
            activations.append(act.cpu().numpy())

        activations = np.concatenate(activations, axis=0)
        return activations


def list_image_paths(folder: str):
    exts = (".png", ".jpg", ".jpeg")
    return sorted(
        [
            os.path.join(folder, f)
            for f in os.listdir(folder)
            if f.lower().endswith(exts)
        ]
    )


def compute_mean_cov(feats: np.ndarray):
    mu = np.mean(feats, axis=0)
    sigma = np.cov(feats, rowvar=False)
    return mu, sigma


def frechet_distance(mu1, sigma1, mu2, sigma2):
    from scipy import linalg

    diff = mu1 - mu2
    covmean = linalg.sqrtm(sigma1.dot(sigma2))

    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = diff.dot(diff) + np.trace(sigma1 + sigma2 - 2 * covmean)
    return float(fid)


def compute_fid_for_folders(
    real_dir: str,
    gen_dir: str,
    device: torch.device,
    batch_size: int = 32,
) -> float:
    print("[INFO] Computing FID between:")
    print(f"       real_dir = {real_dir}")
    print(f"       gen_dir  = {gen_dir}")

    real_paths = list_image_paths(real_dir)
    gen_paths = list_image_paths(gen_dir)

    print(f"[INFO] Found {len(real_paths)} real images.")
    print(f"[INFO] Found {len(gen_paths)} generated images.")

    extractor = InceptionFeatureExtractor(device=device)

    print("[DEBUG] Extracting REAL features...")
    real_feats = extractor.get_activations(
        real_paths, batch_size=batch_size, debug=True
    )

    print("[DEBUG] Extracting GENERATED features...")
    gen_feats = extractor.get_activations(gen_paths, batch_size=batch_size, debug=True)

    mu_real, sigma_real = compute_mean_cov(real_feats)
    mu_gen, sigma_gen = compute_mean_cov(gen_feats)

    fid = frechet_distance(mu_real, sigma_real, mu_gen, sigma_gen)
    print(f"[RESULT] FID = {fid:.4f}")

    return fid


def eval_fid(
    real_dir: str,
    ldm_dir: str,
    gldm_dir: str,
    device: torch.device,
    out_json: str,
) -> dict:
    ensure_dir(os.path.dirname(out_json))

    print("\n[=== FID EVALUATION LDM ===]")
    fid_ldm = compute_fid_for_folders(real_dir, ldm_dir, device=device)

    print("\n[=== FID EVALUATION GLDM ===]")
    fid_gldm = compute_fid_for_folders(real_dir, gldm_dir, device=device)

    metrics = {"fid_ldm": fid_ldm, "fid_gldm": fid_gldm}
    save_metrics_dict(out_json, metrics)

    print("\n[=== FINAL FID RESULTS ===]")
    print(metrics)

    return metrics
