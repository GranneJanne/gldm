from training.trainer_ldm import LatentDiffusionTrainer
from training.trainer_vae import VAETrainer

from utils import load_config, parse_args, set_seed


def main():
    args = parse_args()
    cfg = load_config(args.config)
    run_type = cfg.get("type", "")
    set_seed(cfg.get("seed", 42))

    if run_type == "vae":
        trainer = VAETrainer(cfg)
    elif run_type in ("ldm", "gldm", "random"):
        trainer = LatentDiffusionTrainer(cfg)
    else:
        raise ValueError(
            f"Unknown run type in config: {run_type} "
            "(expected 'vae', 'ldm', 'gldm' or 'random')"
        )

    trainer.train()


if __name__ == "__main__":
    main()
