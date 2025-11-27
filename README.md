# GLDM

A latent diffusion model (LDM) trains its noise predictor with a squared error that treats every latent dimension the same. The VAE decoder does not. Moving the latent along some dimensions changes the decoded image a lot, along others hardly at all. GLDM weights each latent dimension of the epsilon-prediction loss by the matching diagonal entry of the decoder's pullback metric $G(z) = J(z)^\top J(z)$, evaluated at the clean latent.

## Results

CelebA-64, two epochs of training, FID between 5,000 samples and the first 5,000 test images at 64x64, mean and standard deviation over five seeds. Lower is better.

| Model                 | Weights in the loss                                 | FID          |
| --------------------- | --------------------------------------------------- | ------------ |
| LDM                   | none                                                | 80.28 ± 1.58 |
| Random-weight control | positive random weights, same normalisation as GLDM | 80.05 ± 1.33 |
| GLDM                  | square root of the normalised decoder metric        | 76.70 ± 0.98 |

GLDM has the lowest FID in every seed. Random weights give almost no gain, so the improvement comes from the decoder geometry. With 20 epochs of training the baselines catch up, so the advantage of GLDM is faster training.

## Running

Install the dependencies with `pip install -r requirements.txt`. The CelebA loader does not download the dataset, put it under `data/celeba`.

Train the VAE, then the three diffusion models on the frozen VAE.

```
python train.py --config configs/celeba/vae.yml
python train.py --config configs/celeba/ldm.yml
python train.py --config configs/celeba/gldm.yml
python train.py --config configs/celeba/random.yml
```

Evaluate GLDM (or random) against the LDM. FID needs a folder of real CelebA images, and the paper uses the first 5,000 test images preprocessed like the training data. The paper's numbers are averages over seeds 42 to 46, set with `seed` in each config.

```
python eval.py \
  --ldm_config configs/celeba/ldm.yml \
  --ldm_ckpt experiments/checkpoints/ldm_celebA/ldm_epoch2.pt \
  --gldm_config configs/celeba/gldm.yml \
  --gldm_ckpt experiments/checkpoints/gldm_celebA/gldm_epoch2.pt \
  --num_samples 5000 \
  --fid_real_dir path/to/real_celeba_images \
  --outdir results/eval
```

## Author

William Janowsky. The paper is in [docs/paper.pdf](docs/paper.pdf).
