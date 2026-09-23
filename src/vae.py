"""
Convolutional VAE for 28x28 grayscale images, with configurable latent
dimensionality (used directly by Experiment 1 -- the dimensionality study).

The KL term below implements EXACTLY the formula verified in
verify/verify_math.py::kl_diag_gaussian_to_standard_normal (same
-0.5*sum(1+logvar-mu^2-exp(logvar)) closed form) -- just written with torch
ops instead of numpy ops, so the verified math and the trained math are the
same formula, not just similar.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 4, stride=2, padding=1), nn.ReLU(inplace=True),   # 28->14
            nn.Conv2d(32, 64, 4, stride=2, padding=1), nn.ReLU(inplace=True),  # 14->7
        )
        self.fc_mu = nn.Linear(64 * 7 * 7, latent_dim)
        self.fc_logvar = nn.Linear(64 * 7 * 7, latent_dim)

    def forward(self, x):
        h = self.conv(x).flatten(1)
        return self.fc_mu(h), self.fc_logvar(h)


class Decoder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 64 * 7 * 7)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1), nn.ReLU(inplace=True),  # 7->14
            nn.ConvTranspose2d(32, 1, 4, stride=2, padding=1), nn.Tanh(),               # 14->28, output in [-1,1]
        )

    def forward(self, z):
        h = self.fc(z).view(-1, 64, 7, 7)
        return self.deconv(h)


class VAE(nn.Module):
    def __init__(self, latent_dim=16):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decoder(z)
        return x_hat, mu, logvar, z


def kl_divergence(mu, logvar):
    """Same closed form as diffusion_utils.kl_diag_gaussian_to_standard_normal,
    verified with NumPy; summed over latent dims, per-sample."""
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)


def vae_loss(x, x_hat, mu, logvar, beta=1.0):
    recon = F.mse_loss(x_hat, x, reduction='none').flatten(1).sum(-1)  # per-sample
    kl = kl_divergence(mu, logvar)
    loss = (recon + beta * kl).mean()
    return loss, recon.mean(), kl.mean()
