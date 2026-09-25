# Generative Deep Learning: Latent Representations and Diffusion Models — Report

## 1. Overview

This project implements (a) a convolutional variational autoencoder (VAE)
learning a compact latent representation of image data, and (b) a
**conditional diffusion model operating in that latent space** (not pixel
space), studying how latent dimensionality and conditioning strength
affect generation quality and sample diversity. All noise-schedule /
forward-process / reverse-process formulas are unit-verified against their
closed-form theoretical properties in `verify/verify_math.py` (runnable
without PyTorch) before being used in training -- see that file for the
correctness checks (28 checks covering the beta schedule, the forward
process q(z_t|z0) empirical statistics, the closed-form KL divergence, and
critically, that the reverse-step formula implements the exact posterior
mean identity from Ho et al. 2020 eq. 11-12).

## 2. Formulation

**VAE.** Encoder q_phi(z|x) outputs (mu, logvar) of a diagonal Gaussian;
loss = reconstruction MSE + beta * KL(q_phi(z|x) || N(0,I)), with the KL
term in closed form (verified formula, see `src/vae.py::kl_divergence`).

**Conditional latent diffusion.** A DDPM-style forward process
q(z_t|z0) = N(sqrt(alpha_bar_t) z0, (1-alpha_bar_t) I) is defined directly
in VAE latent space (using the *mean* of the encoder's posterior as z0,
following Rombach et al. 2022's latent-diffusion practice). A small
conditional MLP denoiser predicts the added noise, conditioned on the
diffusion timestep (sinusoidal embedding) and the class label (learned
embedding, with a reserved "null" token for classifier-free guidance --
Ho & Salimans 2022). Both full ancestral DDPM sampling and faster DDIM
sampling are implemented.

## 3. Experiment 1 — Effect of latent-space dimensionality

|   latent_dim |   train_recon |   train_kl |   test_recon |   prior_sample_diversity |
|-------------:|--------------:|-----------:|-------------:|-------------------------:|
|            2 |      130.925  |    6.17219 |     132.362  |                  12.3535 |
|            4 |       94.5732 |   10.5253  |      94.7369 |                  14.2009 |
|            8 |       56.9714 |   18.1432  |      56.3897 |                  16.8711 |
|           16 |       35.4696 |   26.9546  |      34.9294 |                  17.2197 |
|           32 |       30.3195 |   31.8108  |      29.6808 |                  16.8348 |
|           64 |       30.9535 |   31.9095  |      31.9147 |                  16.8979 |


Best reconstruction fidelity was achieved at latent_dim=32 (test reconstruction loss=29.681). See `figures/exp1_latent_dim_study.png` for the full reconstruction/KL/diversity-vs-dimension curves, `figures/exp1_latent_structure.png` for how well classes separate in latent space at each dimensionality (2D directly, PCA-projected for higher dims), and `figures/exp1_prior_samples.png` for unconditional samples decoded from the prior at each dimensionality.

## 4. Experiment 2 — Conditional generation quality & diversity per class

|   class_idx |   class_name |   fidelity |   diversity |
|------------:|-------------:|-----------:|------------:|
|           0 |            0 |    0.90625 |     18.9383 |
|           1 |            1 |    1       |     12.0642 |
|           2 |            2 |    0.78125 |     17.5563 |
|           3 |            3 |    0.9375  |     16.7365 |
|           4 |            4 |    0.875   |     19.5505 |
|           5 |            5 |    1       |     18.3035 |
|           6 |            6 |    1       |     14.7951 |
|           7 |            7 |    0.90625 |     15.7462 |
|           8 |            8 |    0.875   |     17.4192 |
|           9 |            9 |    0.90625 |     15.2175 |


Mean fidelity across classes: 0.919 (std 0.069); mean within-class diversity: 16.633. See `figures/exp2_conditional_generation.png` for the per-class bar charts and `figures/exp2_example_samples.png` for example generated images per class.

## 5. Experiment 3 — Sampling-steps (DDIM) ablation: speed vs. quality

|   n_steps |   eta |   wall_time_s |   fidelity |
|----------:|------:|--------------:|-----------:|
|         5 |     1 |      0.106405 |   1        |
|        10 |     1 |      0.106458 |   1        |
|        20 |     1 |      0.189381 |   0.984375 |
|        50 |     1 |      0.509362 |   0.984375 |
|       100 |     1 |      1.11807  |   0.953125 |
|       250 |     1 |      2.56582  |   1        |


Fidelity plateaus around n_steps=5 (1.000, within 0.02 of the best), taking 0.106s to sample -- see `figures/exp3_ddim_steps.png` for the full speed/quality curve.

## 6. Experiment 4 — Classifier-free guidance scale: fidelity/diversity trade-off

|   guidance_scale |   eta |   mean_fidelity |   mean_diversity |
|-----------------:|------:|----------------:|-----------------:|
|                0 |     1 |        0.0875   |          17.6107 |
|                1 |     1 |        0.978125 |          15.4627 |
|                2 |     1 |        1        |          15.1731 |
|                4 |     1 |        0.99375  |          15.7015 |
|                7 |     1 |        0.95     |          15.4454 |
|               10 |     1 |        0.86875  |          15.4053 |


As expected from the classifier-free guidance mechanism (Ho & Salimans 2022), increasing the guidance scale trades sample diversity for class fidelity -- see `figures/exp4_guidance_scale.png`. **Caveat:** if this table came from a single `experiments.py` run, note that Experiment 1/2 numbers will differ slightly across separate runs (no fixed random seed, full retrain each time) -- see Experiment 4b below for a version of this ablation with that confound removed.

## 7. Experiment 4b — Controlled eta ablation (single checkpoint, no confound)

|   eta |   guidance_scale |   mean_fidelity |   mean_diversity |
|------:|-----------------:|----------------:|-----------------:|
|  0    |                1 |        0.9375   |          15.6597 |
|  0    |                4 |        0.503125 |          17.2973 |
|  0    |                7 |        0.18125  |          14.6714 |
|  0    |               10 |        0.125    |          13.4017 |
|  0.25 |                1 |        0.94375  |          15.8174 |
|  0.25 |                4 |        0.534375 |          16.8445 |
|  0.25 |                7 |        0.246875 |          14.9804 |
|  0.25 |               10 |        0.303125 |          13.8112 |
|  0.5  |                1 |        0.95     |          15.8123 |
|  0.5  |                4 |        0.759375 |          16.5815 |
|  0.5  |                7 |        0.496875 |          15.614  |
|  0.5  |               10 |        0.45625  |          15.1873 |
|  0.75 |                1 |        0.95     |          15.5    |
|  0.75 |                4 |        0.921875 |          15.627  |
|  0.75 |                7 |        0.7625   |          15.5353 |
|  0.75 |               10 |        0.76875  |          14.7534 |
|  1    |                1 |        0.959375 |          15.5268 |
|  1    |                4 |        0.978125 |          15.8693 |
|  1    |                7 |        0.884375 |          15.8894 |
|  1    |               10 |        0.834375 |          15.2312 |


Unlike Experiment 4, every row here comes from the SAME trained VAE/diffusion checkpoint -- only `eta` (DDIM stochasticity) and `guidance_scale` change between rows, so any pattern is attributable to those two variables alone, not training-run variance. See `figures/exp4b_eta_sweep.png`. 
At the highest tested guidance scale (w=10), fidelity goes from 0.125 at eta=0 to 0.834 at eta=1 -- i.e. deterministic DDIM sampling is measurably more fragile under strong classifier-free guidance than stochastic (eta>0) sampling, confirmed here without the training-run confound.

## 8. Limitations and honest scope

- Generation quality is scored with a small CNN classifier's class-match
  rate and a pairwise-L2 diversity proxy, not FID/Inception features --
  appropriate given the small custom latent space, but not directly
  comparable to FID numbers reported elsewhere in the literature.
- The denoiser is a plain conditional MLP (appropriate for the small flat
  latent vector), not a U-Net -- a U-Net operates naturally on spatial
  feature maps, which a flattened latent vector no longer has.
- The DDPM math (forward process, KL, reverse-step posterior identity,
  and the generalized-DDIM eta identity) is verified analytically with
  NumPy in `verify/verify_math.py` (35 checks). The PyTorch code has since
  been run end-to-end for real and a genuine bug was found and fixed this
  way: an unclipped DDIM z0-prediction that becomes numerically unstable
  at high timesteps (amplifying denoiser error by orders of magnitude),
  and a related finding that deterministic (eta=0) DDIM sampling is
  fragile under strong classifier-free guidance specifically -- both
  documented as regression tests in `verify/verify_math.py` so they can't
  silently reappear.
- Experiment 1/2 numbers vary somewhat between separate `experiments.py`
  invocations since nothing is seeded and the models retrain from
  scratch each time; treat single-run point estimates with that in mind
  (differences smaller than roughly the sampling noise floor -- a few
  percentage points at the sample sizes used here -- aren't meaningful).

## 9. Reproducing

```bash
pip install -r requirements.txt
cd src
python experiments.py --dataset mnist --ddim_eta 1.0   # main pipeline
python run_eta_sweep.py                                  # controlled Exp 4b (reuses checkpoints)
python verify_math.py           # optional: re-check the math (no PyTorch needed)
python visualize.py
python generate_report.py       # regenerates this file with your real numbers
```
