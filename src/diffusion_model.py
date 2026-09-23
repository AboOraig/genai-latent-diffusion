"""
Conditional diffusion model operating in the VAE's latent space (not pixel
space -- this is what makes it a *latent* diffusion model: much cheaper to 
train than pixel-space diffusion since the latent
dimensionality is small, e.g. 2-64, vs. 784 raw pixels).

All noise-schedule / forward-process / reverse-step formulas here are the
EXACT formulas verified (with NumPy) in verify/verify_math.py -- see the
docstring of each for the cross-reference. Class conditioning uses
classifier-free guidance (Ho & Salimans, 2022): during training the class
label is replaced with a learned "null" token with probability
`uncond_prob`; at sampling time the conditional and unconditional
predictions are combined with a guidance scale `w`.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from diffusion_utils import linear_beta_schedule, cosine_beta_schedule, compute_alphas


def sinusoidal_time_embedding_torch(t, dim, device):
    """Torch port of diffusion_utils.sinusoidal_time_embedding -- SAME
    formula (half sin, half cos, log-spaced frequencies), verified with
    NumPy; only the tensor library differs."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000) * torch.arange(half, device=device, dtype=torch.float32) / max(half - 1, 1)
    )
    args = t[:, None].float() * freqs[None, :]
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2 == 1:
        emb = torch.cat([emb, torch.zeros(emb.shape[0], 1, device=device)], dim=-1)
    return emb


class ResidualBlock(nn.Module):
    def __init__(self, dim, cond_dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.cond_proj = nn.Linear(cond_dim, dim)

    def forward(self, x, cond):
        h = self.norm(x)
        h = F.silu(self.fc1(h) + self.cond_proj(cond))
        h = self.fc2(h)
        return x + h


class Denoiser(nn.Module):
    """Small conditional MLP predicting the noise added to a latent vector.
    Latent dims here are small (<=64), so an MLP is appropriate -- a
    convolutional U-Net (standard for pixel-space diffusion) would be
    overkill and structurally meaningless on a flat latent vector."""

    def __init__(self, latent_dim, num_classes, hidden=256, time_dim=64, n_blocks=4):
        super().__init__()
        self.time_dim = time_dim
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, hidden), nn.SiLU(), nn.Linear(hidden, hidden)
        )
        # +1 class slot reserved as the "null" / unconditional token for CFG
        self.class_embed = nn.Embedding(num_classes + 1, hidden)
        self.in_proj = nn.Linear(latent_dim, hidden)
        self.blocks = nn.ModuleList([ResidualBlock(hidden, hidden) for _ in range(n_blocks)])
        self.out_proj = nn.Linear(hidden, latent_dim)

    def forward(self, z_t, t, y):
        """z_t: [B, latent_dim], t: [B] long, y: [B] long (class idx, or
        num_classes for the null token)."""
        t_emb = sinusoidal_time_embedding_torch(t, self.time_dim, z_t.device)
        t_emb = self.time_mlp(t_emb)
        y_emb = self.class_embed(y)
        cond = t_emb + y_emb
        h = self.in_proj(z_t)
        for block in self.blocks:
            h = block(h, cond)
        return self.out_proj(h)


class LatentDiffusion:
    """Wraps the noise schedule (verified formulas) + denoiser network into
    training-loss and CFG-sampling methods."""

    def __init__(self, denoiser, num_classes, T=1000, schedule='cosine', device='cpu',
                 latent_mean=None, latent_std=None, clip_std_mult=4.0):
        self.denoiser = denoiser
        self.num_classes = num_classes
        self.null_token = num_classes  # reserved index in Denoiser.class_embed
        self.T = T
        self.device = device
        self.clip_std_mult = clip_std_mult

        betas_np = cosine_beta_schedule(T) if schedule == 'cosine' else linear_beta_schedule(T)
        alphas_np, alpha_bars_np = compute_alphas(betas_np)
        self.betas = torch.tensor(betas_np, dtype=torch.float32, device=device)
        self.alphas = torch.tensor(alphas_np, dtype=torch.float32, device=device)
        self.alpha_bars = torch.tensor(alpha_bars_np, dtype=torch.float32, device=device)

        # Data-driven bounds for clipping the predicted z0 inside DDIM steps.
        # NECESSARY, not cosmetic: as t -> T, alpha_bar_t -> 0, so
        # z0_pred = (z_t - sqrt(1-abar)*eps_pred)/sqrt(abar) divides by a
        # near-zero number and amplifies any small denoiser error by
        # orders of magnitude (verified numerically: ~20,000x at t=999 for
        # this schedule) -- exactly the mechanism that breaks unclipped
        # DDIM even with n_steps==T (no step-skipping at all). Every
        # production DDIM implementation (Ho et al.'s own code, HF
        # diffusers' `clip_sample`) guards against this; we do the same,
        # but with bounds derived from the real training-latent
        # distribution rather than an arbitrary constant, since this is
        # latent-space (not pixel [-1,1]) diffusion.
        if latent_mean is not None and latent_std is not None:
            self.latent_min = torch.tensor(
                latent_mean - clip_std_mult * latent_std, dtype=torch.float32, device=device)
            self.latent_max = torch.tensor(
                latent_mean + clip_std_mult * latent_std, dtype=torch.float32, device=device)
        else:
            self.latent_min, self.latent_max = None, None
            print("WARNING: LatentDiffusion built without latent_mean/latent_std -- "
                  "sample_ddim will NOT clip z0_pred and may be numerically unstable "
                  "at high timesteps. Pass latent statistics from training set.")

    def q_sample(self, z0, t, noise):
        """Identical formula to diffusion_utils.q_sample (verified)."""
        sqrt_ab = self.alpha_bars[t].sqrt()[:, None]
        sqrt_1m_ab = (1 - self.alpha_bars[t]).sqrt()[:, None]
        return sqrt_ab * z0 + sqrt_1m_ab * noise

    def training_loss(self, z0, y, uncond_prob=0.1):
        B = z0.shape[0]
        t = torch.randint(0, self.T, (B,), device=self.device)
        noise = torch.randn_like(z0)
        z_t = self.q_sample(z0, t, noise)

        # classifier-free guidance: randomly drop the label to the null token
        drop_mask = torch.rand(B, device=self.device) < uncond_prob
        y_input = y.clone()
        y_input[drop_mask] = self.null_token

        eps_pred = self.denoiser(z_t, t, y_input)
        return F.mse_loss(eps_pred, noise)

    @torch.no_grad()
    def sample(self, n_samples, y, guidance_scale=1.0, latent_dim=None):
        """Ancestral DDPM sampling with classifier-free guidance.
        y: [n_samples] long tensor of desired class labels.
        guidance_scale w: eps = eps_uncond + w*(eps_cond - eps_uncond).
        w=1.0 recovers plain conditional sampling (no guidance boost);
        w=0.0 is fully unconditional."""
        latent_dim = latent_dim or self.denoiser.out_proj.out_features
        z = torch.randn(n_samples, latent_dim, device=self.device)
        y_null = torch.full_like(y, self.null_token)

        for t_int in reversed(range(self.T)):
            t = torch.full((n_samples,), t_int, device=self.device, dtype=torch.long)
            eps_cond = self.denoiser(z, t, y)
            if guidance_scale != 1.0:
                eps_uncond = self.denoiser(z, t, y_null)
                eps = eps_uncond + guidance_scale * (eps_cond - eps_uncond)
            else:
                eps = eps_cond

            alpha_t = self.alphas[t_int]
            beta_t = self.betas[t_int]
            alpha_bar_t = self.alpha_bars[t_int]
            coef = beta_t / torch.sqrt(1 - alpha_bar_t)
            mean = (z - coef * eps) / torch.sqrt(alpha_t)

            if t_int > 0:
                noise = torch.randn_like(z)
                z = mean + torch.sqrt(beta_t) * noise
            else:
                z = mean
        return z

    @torch.no_grad()
    def sample_ddim(self, n_samples, y, guidance_scale=1.0, latent_dim=None, n_steps=50):
        """Faster deterministic(-ish) DDIM sampling, subsampling the T-step
        schedule down to n_steps -- used for the speed/quality ablation
        (Experiment 3)."""
        latent_dim = latent_dim or self.denoiser.out_proj.out_features
        z = torch.randn(n_samples, latent_dim, device=self.device)
        y_null = torch.full_like(y, self.null_token)

        step_indices = torch.linspace(0, self.T - 1, n_steps, device=self.device).long()
        step_indices = torch.unique(step_indices, sorted=True).flip(0)  # descending

        for i, t_int in enumerate(step_indices.tolist()):
            t = torch.full((n_samples,), t_int, device=self.device, dtype=torch.long)
            eps_cond = self.denoiser(z, t, y)
            if guidance_scale != 1.0:
                eps_uncond = self.denoiser(z, t, y_null)
                eps = eps_uncond + guidance_scale * (eps_cond - eps_uncond)
            else:
                eps = eps_cond

            alpha_bar_t = self.alpha_bars[t_int]
            z0_pred = (z - torch.sqrt(1 - alpha_bar_t) * eps) / torch.sqrt(alpha_bar_t)
            if self.latent_min is not None:
                z0_pred = torch.clamp(z0_pred, self.latent_min, self.latent_max)

            if i + 1 < len(step_indices):
                t_prev = step_indices[i + 1].item()
                alpha_bar_prev = self.alpha_bars[t_prev]
            else:
                alpha_bar_prev = torch.tensor(1.0, device=self.device)

            # DDIM deterministic update (eta=0)
            z = torch.sqrt(alpha_bar_prev) * z0_pred + torch.sqrt(1 - alpha_bar_prev) * eps
        return z
