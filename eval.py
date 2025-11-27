import argparse
import os

import torch

from eval.utils import (
    load_config,
    build_vae_from_config,
    build_eps_model_from_config,
    build_schedule_from_config,
    get_latent_dim,
    ensure_dir,
    save_metrics_dict,
)
from eval.sample_eval import generate_samples, generate_compare_grid
from eval.fid_kid import eval_fid
from eval.interp_eval import latent_interpolation_grid
from diffusion.sampling import sample_latents_ddpm


def main():
    parser = argparse.ArgumentParser(description="Evaluate LDM vs GLDM")
    parser.add_argument("--ldm_config", type=str, required=True)
    parser.add_argument("--ldm_ckpt", type=str, required=True)
    parser.add_argument("--gldm_config", type=str, required=True)
    parser.add_argument("--gldm_ckpt", type=str, required=True)
    parser.add_argument("--outdir", type=str, default="results/eval")
    parser.add_argument("--num_samples", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--interp_steps", type=int, default=8)
    parser.add_argument("--num_interps", type=int, default=5)
    parser.add_argument("--fid_real_dir", type=str, default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ensure_dir(args.outdir)

    ldm_cfg = load_config(args.ldm_config)
    gldm_cfg = load_config(args.gldm_config)

    vae = build_vae_from_config(ldm_cfg, device=device)
    latent_dim = get_latent_dim(vae)

    eps_ldm = build_eps_model_from_config(
        ldm_cfg, latent_dim=latent_dim, device=device, ckpt_path=args.ldm_ckpt
    )
    eps_gldm = build_eps_model_from_config(
        gldm_cfg, latent_dim=latent_dim, device=device, ckpt_path=args.gldm_ckpt
    )

    schedule_ldm = build_schedule_from_config(ldm_cfg)
    schedule_gldm = build_schedule_from_config(gldm_cfg)

    samples_dir = os.path.join(args.outdir, "samples")
    ensure_dir(samples_dir)

    print("Generating LDM samples...")
    res_ldm = generate_samples(
        vae=vae,
        eps_model=eps_ldm,
        schedule=schedule_ldm,
        num_samples=args.num_samples,
        latent_dim=latent_dim,
        device=device,
        batch_size=args.batch_size,
        outdir=samples_dir,
        prefix="ldm",
        save_individual=True,
    )

    print("Generating GLDM samples...")
    res_gldm = generate_samples(
        vae=vae,
        eps_model=eps_gldm,
        schedule=schedule_gldm,
        num_samples=args.num_samples,
        latent_dim=latent_dim,
        device=device,
        batch_size=args.batch_size,
        outdir=samples_dir,
        prefix="gldm",
        save_individual=True,
    )

    print("Generating comparison grid...")
    compare_grid_path = generate_compare_grid(
        vae=vae,
        eps_model_ldm=eps_ldm,
        eps_model_gldm=eps_gldm,
        schedule_ldm=schedule_ldm,
        schedule_gldm=schedule_gldm,
        num_samples=min(16, args.num_samples),
        latent_dim=latent_dim,
        device=device,
        outdir=samples_dir,
        nrow=4,
    )

    metrics = {}
    if args.fid_real_dir is not None:
        print("Computing FID...")
        fid_json = os.path.join(args.outdir, "fid.json")
        fid_metrics = eval_fid(
            real_dir=args.fid_real_dir,
            ldm_dir=os.path.join(samples_dir, "ldm_samples"),
            gldm_dir=os.path.join(samples_dir, "gldm_samples"),
            device=device,
            out_json=fid_json,
        )
        metrics.update(fid_metrics)
    else:
        print("Skipping FID (no --fid_real_dir provided)")

    interp_dir = os.path.join(args.outdir, "interp")
    ensure_dir(interp_dir)

    interp_ldm_all: list[dict[str, float]] = []
    interp_gldm_all: list[dict[str, float]] = []

    for i in range(args.num_interps):
        print(f"Interpolation run {i + 1}/{args.num_interps}")

        if i <= args.num_interps // 2:
            print("  Sampling endpoints (LDM prior)...")
            zA = sample_latents_ddpm(
                eps_model=eps_ldm,
                schedule=schedule_ldm,
                num_samples=1,
                latent_dim=latent_dim,
                device=device,
                z_T=None,
            )  # [1, d]
            zB = sample_latents_ddpm(
                eps_model=eps_ldm,
                schedule=schedule_ldm,
                num_samples=1,
                latent_dim=latent_dim,
                device=device,
                z_T=None,
            )  # [1, d]
        else:
            print("  Sampling endpoints (GLDM prior)...")
            zA = sample_latents_ddpm(
                eps_model=eps_gldm,
                schedule=schedule_gldm,
                num_samples=1,
                latent_dim=latent_dim,
                device=device,
                z_T=None,
            )
            zB = sample_latents_ddpm(
                eps_model=eps_gldm,
                schedule=schedule_gldm,
                num_samples=1,
                latent_dim=latent_dim,
                device=device,
                z_T=None,
            )

        print("  Evaluating interpolation (LDM)...")
        interp_ldm_path = os.path.join(interp_dir, f"interp_ldm_{i:03d}.png")
        interp_ldm_metrics = latent_interpolation_grid(
            vae=vae,
            eps_model=eps_ldm,
            schedule=schedule_ldm,
            latent_dim=latent_dim,
            device=device,
            steps=args.interp_steps,
            out_path=interp_ldm_path,
            zA=zA,
            zB=zB,
        )
        interp_ldm_all.append(interp_ldm_metrics)

        print("  Evaluating interpolation (GLDM)...")
        interp_gldm_path = os.path.join(interp_dir, f"interp_gldm_{i:03d}.png")
        interp_gldm_metrics = latent_interpolation_grid(
            vae=vae,
            eps_model=eps_gldm,
            schedule=schedule_gldm,
            latent_dim=latent_dim,
            device=device,
            steps=args.interp_steps,
            out_path=interp_gldm_path,
            zA=zA,
            zB=zB,
        )
        interp_gldm_all.append(interp_gldm_metrics)

    def aggregate_metric_list(metric_dicts: list[dict[str, float]]) -> dict[str, float]:
        """
        Take a list like:
            [{'lpips': 0.10, 'l2': 0.01}, {'lpips': 0.12, 'l2': 0.011}, ...]
        and return mean per key, ignoring missing keys and NaNs.
        """
        import math

        acc: dict[str, list[float]] = {}
        for d in metric_dicts:
            for k, v in d.items():
                if v is None or math.isnan(v):
                    continue
                acc.setdefault(k, []).append(float(v))

        out: dict[str, float] = {}
        for k, vals in acc.items():
            if len(vals) == 0:
                out[k] = float("nan")
            else:
                out[k] = float(sum(vals) / len(vals))
        return out

    interp_ldm_mean = aggregate_metric_list(interp_ldm_all)
    interp_gldm_mean = aggregate_metric_list(interp_gldm_all)

    interp_summary = {
        "ldm": {
            "per_interp": interp_ldm_all,
            "mean": interp_ldm_mean,
        },
        "gldm": {
            "per_interp": interp_gldm_all,
            "mean": interp_gldm_mean,
        },
    }

    save_metrics_dict(os.path.join(interp_dir, "interp_metrics.json"), interp_summary)

    metrics.update(
        {
            "interp_ldm_lpips": interp_ldm_mean.get("lpips", float("nan")),
            "interp_gldm_lpips": interp_gldm_mean.get("lpips", float("nan")),
        }
    )

    save_metrics_dict(os.path.join(args.outdir, "summary.json"), metrics)

    print("Evaluation complete.")
    print("Summary:", metrics)
    print("Sample grids:", res_ldm["grid"], res_gldm["grid"])
    print("Compare grid:", compare_grid_path)


if __name__ == "__main__":
    main()
