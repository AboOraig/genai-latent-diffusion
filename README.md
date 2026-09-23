# Generative Deep Learning: Latent Representations and Diffusion Models

A VAE learning compact latent representations, extended with a
**conditional diffusion model trained directly in that latent space**
(latent diffusion, à la Rombach et al. 2022), studying how latent
dimensionality and conditioning strength affect generation quality and
sample diversity.

## Quickstart (Colab or any GPU machine)

```bash
git clone <this-repo>
cd genai_project
pip install -r requirements.txt
cd src
python experiments.py --dataset mnist
python visualize.py
python generate_report.py
```

Rough runtime on a free Colab GPU: VAE training (6 latent dims × 15
epochs) ~5-10 min, diffusion training (100 epochs on ~60k tiny latent
vectors) ~5-15 min, the three evaluation experiments (sampling loops)
~5-15 min depending on step counts. Reduce `--epochs_vae` /
`--epochs_diffusion` / the `--latent_dims` list for a faster first pass.

## Structure

```
src/
  data.py                 MNIST / Fashion-MNIST loading (torchvision)
  vae.py                  convolutional VAE, configurable latent dim
  diffusion_utils.py       framework-agnostic diffusion math (VERIFIED, see verify/)
  diffusion_model.py       conditional MLP denoiser + DDPM/DDIM sampling (PyTorch)
  classifier.py             small CNN used only to SCORE generated samples
  train_utils.py            VAE / diffusion training loops
  experiments.py            the four computational studies (run this)
  visualize.py               all figures (run after experiments.py; tested with dummy data)
  generate_report.py         fills REPORT.md with actual results (run last)
verify/
  verify_math.py             NumPy-only correctness checks for the diffusion/VAE math
results/                    CSVs + raw arrays written by experiments.py
figures/                     PNGs written by visualize.py
REPORT.md                    placeholder until run the pipeline
```

## What's implemented

1. **VAE with compact latent representations** (`vae.py`) — convolutional
   encoder/decoder, closed-form KL regularizer (verified formula).
2. **Conditional diffusion model in latent space** (`diffusion_model.py`)
   — DDPM forward process + both full ancestral and fast DDIM sampling,
   classifier-free guidance for conditioning.
3. **Effect of latent dimensionality / conditioning on quality & diversity**
   (`experiments.py`) — four studies: latent-dim sweep, per-class
   conditional-generation fidelity/diversity, DDIM steps speed/quality
   trade-off, and guidance-scale fidelity/diversity trade-off.
4. **Computational experiments with visual results** (`visualize.py`) —
   dimensionality curves, latent-space scatter plots, sample grids, and
   trade-off curves, all from real run data (dummy-data-tested plotting
   code).
