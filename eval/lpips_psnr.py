import torch
import torch.nn.functional as F


try:
    import lpips  # type: ignore

    _HAS_LPIPS = True
except ImportError:
    _HAS_LPIPS = False


def compute_psnr(x: torch.Tensor, y: torch.Tensor) -> float:
    """
    x, y: [B, C, H, W], in [0,1] or [-1,1] (normalized).
    We convert to [0,1] and compute PSNR.
    """
    # assume inputs in [-1, 1]; push to [0, 1]
    x = (x + 1) / 2
    y = (y + 1) / 2
    mse = F.mse_loss(x, y).item()
    if mse == 0:
        return float("inf")
    psnr = 10 * torch.log10(torch.tensor(1.0 / mse))
    return float(psnr.item())


def compute_l2(x: torch.Tensor, y: torch.Tensor) -> float:
    return float(F.mse_loss(x, y).item())


def compute_lpips(x: torch.Tensor, y: torch.Tensor, net: str = "vgg") -> float:
    if not _HAS_LPIPS:
        return float("nan")
    loss_fn = lpips.LPIPS(net=net).to(x.device)
    with torch.no_grad():
        d = loss_fn(x, y)
    return float(d.mean().item())


def compute_pairwise_metrics(x: torch.Tensor, y: torch.Tensor) -> dict[str, float]:
    """Compute a small set of distances between two batches of images."""
    metrics = {
        "l2": compute_l2(x, y),
        "psnr": compute_psnr(x, y),
    }
    if _HAS_LPIPS:
        metrics["lpips"] = compute_lpips(x, y)
    return metrics
