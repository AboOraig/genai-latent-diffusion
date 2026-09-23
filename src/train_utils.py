"""Training loops for the VAE and the conditional latent diffusion model,
kept separate from experiments.py so they can be unit-imported/reused."""
import torch
import numpy as np

from vae import VAE, vae_loss
from diffusion_model import Denoiser, LatentDiffusion


def train_vae(train_loader, test_loader, latent_dim, device, epochs=15, lr=1e-3, beta=1.0):
    model = VAE(latent_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    history = []
    for epoch in range(epochs):
        model.train()
        recon_sum, kl_sum, n = 0.0, 0.0, 0
        for x, _ in train_loader:
            x = x.to(device)
            opt.zero_grad()
            x_hat, mu, logvar, _ = model(x)
            loss, recon, kl = vae_loss(x, x_hat, mu, logvar, beta=beta)
            loss.backward()
            opt.step()
            recon_sum += recon.item() * x.size(0)
            kl_sum += kl.item() * x.size(0)
            n += x.size(0)
        train_recon, train_kl = recon_sum / n, kl_sum / n

        model.eval()
        test_recon = evaluate_vae_recon(model, test_loader, device)
        history.append(dict(epoch=epoch, train_recon=train_recon, train_kl=train_kl,
                             test_recon=test_recon))
        print(f"  [VAE d={latent_dim}] epoch {epoch+1}/{epochs} "
              f"train_recon={train_recon:.2f} train_kl={train_kl:.2f} test_recon={test_recon:.2f}")
    return model, history


@torch.no_grad()
def evaluate_vae_recon(model, loader, device):
    model.eval()
    total, n = 0.0, 0
    for x, _ in loader:
        x = x.to(device)
        x_hat, mu, logvar, _ = model(x)
        recon = torch.nn.functional.mse_loss(x_hat, x, reduction='none').flatten(1).sum(-1)
        total += recon.sum().item()
        n += x.size(0)
    return total / n


@torch.no_grad()
def encode_dataset(vae, loader, device, use_mean=True):
    """Encode an entire dataloader into (latents, labels) arrays. Uses the
    posterior MEAN (not a stochastic sample) as the latent representation
    fed to the diffusion model, following standard latent-diffusion
    practice (Rombach et al. 2022)."""
    vae.eval()
    zs, ys = [], []
    for x, y in loader:
        x = x.to(device)
        mu, logvar = vae.encoder(x)
        z = mu if use_mean else vae.reparameterize(mu, logvar)
        zs.append(z.cpu().numpy())
        ys.append(y.numpy())
    return np.concatenate(zs), np.concatenate(ys)


def train_latent_diffusion(latents, labels, num_classes, device, latent_dim,
                            epochs=100, batch_size=256, lr=2e-4, T=1000,
                            schedule='cosine', uncond_prob=0.1, hidden=256, n_blocks=4,
                            clip_std_mult=4.0):
    denoiser = Denoiser(latent_dim, num_classes, hidden=hidden, n_blocks=n_blocks).to(device)
    # per-dimension mean/std of the REAL encoded training latents -- used to
    # clip DDIM's predicted z0 at each step (see LatentDiffusion docstring
    # in diffusion_model.py for why this is required, not optional).
    latent_mean = latents.mean(axis=0)
    latent_std = latents.std(axis=0)
    diffusion = LatentDiffusion(denoiser, num_classes, T=T, schedule=schedule, device=device,
                                 latent_mean=latent_mean, latent_std=latent_std,
                                 clip_std_mult=clip_std_mult)
    opt = torch.optim.Adam(denoiser.parameters(), lr=lr)

    z_all = torch.tensor(latents, dtype=torch.float32, device=device)
    y_all = torch.tensor(labels, dtype=torch.long, device=device)
    n = z_all.shape[0]

    history = []
    for epoch in range(epochs):
        perm = torch.randperm(n, device=device)
        loss_sum = 0.0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            z0, y = z_all[idx], y_all[idx]
            opt.zero_grad()
            loss = diffusion.training_loss(z0, y, uncond_prob=uncond_prob)
            loss.backward()
            opt.step()
            loss_sum += loss.item() * z0.size(0)
        avg_loss = loss_sum / n
        history.append(dict(epoch=epoch, loss=avg_loss))
        if (epoch + 1) % max(1, epochs // 10) == 0 or epoch == 0:
            print(f"  [diffusion d={latent_dim}] epoch {epoch+1}/{epochs} loss={avg_loss:.4f}")
    return diffusion, history
