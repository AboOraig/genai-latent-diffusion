"""
Pure math for the DDPM-style latent diffusion process, deliberately kept
framework-agnostic (plain array arithmetic that works identically on numpy
arrays or torch tensors) so that:
  (a) it can be unit-verified with NumPy alone (see verify/verify_math.py),
      without needing PyTorch installed, and
  (b) the exact same formulas are reused, unmodified, inside the PyTorch
      training/sampling code (src/diffusion_model.py) -- so verifying this
      file *is* verifying the math actually used at train/sample time.

Reference: Ho, Jain & Abbeel, "Denoising Diffusion Probabilistic Models"
(2020), and Nichol & Dhariwal, "Improved DDPM" (2021) for the cosine
schedule.
"""
import numpy as np


def linear_beta_schedule(T, beta_start=1e-4, beta_end=0.02):
    """Standard linear variance schedule from Ho et al. 2020."""
    return np.linspace(beta_start, beta_end, T)


def cosine_beta_schedule(T, s=0.008):
    """Cosine schedule from Nichol & Dhariwal 2021 -- smoother SNR decay,
    generally gives better sample quality than the linear schedule."""
    steps = T + 1
    x = np.linspace(0, T, steps)
    alphas_bar = np.cos(((x / T) + s) / (1 + s) * np.pi / 2) ** 2
    alphas_bar = alphas_bar / alphas_bar[0]
    betas = 1 - (alphas_bar[1:] / alphas_bar[:-1])
    return np.clip(betas, 1e-8, 0.999)


def compute_alphas(betas):
    """Returns (alphas, alpha_bars) where alpha_t = 1-beta_t and
    alpha_bar_t = prod_{s<=t} alpha_s."""
    alphas = 1.0 - betas
    alpha_bars = np.cumprod(alphas)
    return alphas, alpha_bars


def q_sample(z0, t, alpha_bars, noise):
    """Closed-form forward diffusion: sample z_t ~ q(z_t | z_0) directly,
    without iterating through all intermediate steps.

        z_t = sqrt(alpha_bar_t) * z0 + sqrt(1 - alpha_bar_t) * noise

    `t` is an integer array of per-sample timesteps; alpha_bars is indexed
    by t. Works for both numpy arrays and torch tensors given matching
    indexing/broadcasting semantics.
    """
    sqrt_ab = np.sqrt(alpha_bars[t])
    sqrt_1m_ab = np.sqrt(1.0 - alpha_bars[t])
    # broadcast over feature dims
    while sqrt_ab.ndim < z0.ndim:
        sqrt_ab = sqrt_ab[..., None]
        sqrt_1m_ab = sqrt_1m_ab[..., None]
    return sqrt_ab * z0 + sqrt_1m_ab * noise


def predict_z0_from_noise(z_t, t, alpha_bars, eps_pred):
    """Invert q_sample to recover the model's implicit z0 estimate from its
    predicted noise -- used both for diagnostics and inside the reverse
    (sampling) step."""
    sqrt_ab = np.sqrt(alpha_bars[t])
    sqrt_1m_ab = np.sqrt(1.0 - alpha_bars[t])
    while sqrt_ab.ndim < z_t.ndim:
        sqrt_ab = sqrt_ab[..., None]
        sqrt_1m_ab = sqrt_1m_ab[..., None]
    return (z_t - sqrt_1m_ab * eps_pred) / sqrt_ab


def ddpm_reverse_step_mean_var(z_t, t, eps_pred, betas, alphas, alpha_bars):
    """One reverse (denoising) step of ancestral DDPM sampling, given the
    model's predicted noise eps_pred at timestep t.

        mu_theta(z_t, t) = 1/sqrt(alpha_t) * ( z_t - beta_t/sqrt(1-alpha_bar_t) * eps_pred )
        Var = beta_t   (the simple fixed-variance choice from Ho et al. 2020)

    `t` is a per-sample integer array (shape [N]); in the standard ancestral
    sampling loop every sample shares the same scalar t at a given step, but
    this also works with per-sample t for generality.

    Returns (mean, var): mean has the same shape as z_t; var has shape [N]
    (broadcast `sqrt(var)[...,None] * noise` when adding it back, and skip
    entirely at t=0 since the final step is noise-free in DDPM).
    """
    alpha_t = alphas[t]
    beta_t = betas[t]
    alpha_bar_t = alpha_bars[t]
    coef = beta_t / np.sqrt(1.0 - alpha_bar_t)
    sqrt_alpha_t = np.sqrt(alpha_t)
    while coef.ndim < z_t.ndim:
        coef = coef[..., None]
        sqrt_alpha_t = sqrt_alpha_t[..., None]
    mean = (z_t - coef * eps_pred) / sqrt_alpha_t
    var = beta_t
    return mean, var


def kl_diag_gaussian_to_standard_normal(mu, logvar):
    """Closed-form KL( N(mu, diag(exp(logvar))) || N(0, I) ), summed over
    the latent dimension, per-sample. Standard VAE regularizer.

        KL = -0.5 * sum( 1 + logvar - mu^2 - exp(logvar) )
    """
    return -0.5 * np.sum(1.0 + logvar - mu ** 2 - np.exp(logvar), axis=-1)


def sinusoidal_time_embedding(t, dim):
    """Transformer-style sinusoidal embedding of integer timesteps `t`
    (shape [B]) into a `dim`-dimensional vector, used to condition the
    denoiser network on the diffusion timestep."""
    half = dim // 2
    freqs = np.exp(-np.log(10000) * np.arange(half) / max(half - 1, 1))
    args = t[:, None].astype(np.float64) * freqs[None, :]
    emb = np.concatenate([np.sin(args), np.cos(args)], axis=-1)
    if dim % 2 == 1:
        emb = np.concatenate([emb, np.zeros((emb.shape[0], 1))], axis=-1)
    return emb
