"""
Diagnostic for the Experiment 3/4 anomaly: DDIM fidelity looks flat/low
regardless of step count, and non-monotonic in guidance scale. This script
reuses ALREADY-TRAINED checkpoints (no retraining) to isolate the cause. 
Run from src/:
    python diagnose_ddim.py

It prints four fidelity numbers for class 0. Read the interpretation guide
at the bottom of this file's output once we have them.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import torch

from vae import VAE
from diffusion_model import Denoiser, LatentDiffusion
from classifier import train_classifier, classification_fidelity
from train_utils import encode_dataset
from data import get_dataloaders

LATENT_DIM = 16      # must match what we trained with (--main_latent_dim default)
NUM_CLASSES = 10
T = 1000
SCHEDULE = 'cosine'
FIXED_CLASS = 0
N_SAMPLES = 64
W = 3.0               # guidance scale used in exp2/exp3


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("device:", device)

    vae = VAE(LATENT_DIM).to(device)
    vae.load_state_dict(torch.load('../results/vae_d16.pt', map_location=device))
    vae.eval()

    denoiser = Denoiser(LATENT_DIM, NUM_CLASSES).to(device)
    denoiser.load_state_dict(torch.load('../results/denoiser.pt', map_location=device))
    denoiser.eval()

    train_loader, test_loader, _ = get_dataloaders('mnist', 128)

    print("\n(re-encoding training set to get latent stats for DDIM clipping -- fast)")
    z_train, _ = encode_dataset(vae, train_loader, device)
    latent_mean, latent_std = z_train.mean(axis=0), z_train.std(axis=0)
    print(f"latent stats: mean range [{latent_mean.min():.2f},{latent_mean.max():.2f}], "
          f"std range [{latent_std.min():.2f},{latent_std.max():.2f}]")

    diffusion = LatentDiffusion(denoiser, NUM_CLASSES, T=T, schedule=SCHEDULE, device=device,
                                 latent_mean=latent_mean, latent_std=latent_std)

    print("\n(re-training a quick classifier for scoring -- ~1 min)")
    classifier = train_classifier(train_loader, test_loader, device, epochs=2)

    y = torch.full((N_SAMPLES,), FIXED_CLASS, dtype=torch.long, device=device)

    print("\n--- Test A: ancestral (full T=1000), w=3.0 [reference: exp2 got 0.844-0.906] ---")
    z = diffusion.sample(N_SAMPLES, y, guidance_scale=W, latent_dim=LATENT_DIM)
    with torch.no_grad():
        imgs = vae.decoder(z)
    fid_a = classification_fidelity(classifier, imgs, y, device)
    print(f"fidelity = {fid_a:.3f}")

    print("\n--- Test B (FIXED): DDIM with n_steps=1000 (no skipping), w=3.0, z0 clipping ON ---")
    z = diffusion.sample_ddim(N_SAMPLES, y, guidance_scale=W, latent_dim=LATENT_DIM, n_steps=1000)
    with torch.no_grad():
        imgs = vae.decoder(z)
    fid_b = classification_fidelity(classifier, imgs, y, device)
    print(f"fidelity = {fid_b:.3f}  (previously 0.562 with the bug)")

    print("\n--- Test E (FIXED): DDIM with n_steps=250, w=3.0, z0 clipping ON [matches original exp3] ---")
    z = diffusion.sample_ddim(N_SAMPLES, y, guidance_scale=W, latent_dim=LATENT_DIM, n_steps=250)
    with torch.no_grad():
        imgs = vae.decoder(z)
    fid_e = classification_fidelity(classifier, imgs, y, device)
    print(f"fidelity = {fid_e:.3f}  (previously ~0.578 with the bug)")

    print("\n--- Test C: DDIM with n_steps=250, w=1.0 (NO guidance), z0 clipping ON ---")
    z = diffusion.sample_ddim(N_SAMPLES, y, guidance_scale=1.0, latent_dim=LATENT_DIM, n_steps=250)
    with torch.no_grad():
        imgs = vae.decoder(z)
    fid_c = classification_fidelity(classifier, imgs, y, device)
    print(f"fidelity = {fid_c:.3f}  (previously 0.203 with the bug)")

    print("\n--- Test D: ancestral (full T=1000), w=1.0 (NO guidance) ---")
    z = diffusion.sample(N_SAMPLES, y, guidance_scale=1.0, latent_dim=LATENT_DIM)
    with torch.no_grad():
        imgs = vae.decoder(z)
    fid_d = classification_fidelity(classifier, imgs, y, device)
    print(f"fidelity = {fid_d:.3f}")

    print("\n--- Test F (NEW): DDIM with n_steps=250, w=3.0, eta=1.0 (ancestral-like robustness) ---")
    z = diffusion.sample_ddim(N_SAMPLES, y, guidance_scale=W, latent_dim=LATENT_DIM,
                               n_steps=250, eta=1.0)
    with torch.no_grad():
        imgs = vae.decoder(z)
    fid_f = classification_fidelity(classifier, imgs, y, device)
    print(f"fidelity = {fid_f:.3f}  (Test E with eta=0.0 was 0.625; A/ancestral reference is ~0.97)")

    print(f"""
=== INTERPRETATION (with the z0-clipping fix applied) ===
A (ancestral, full, w=3.0)          = {fid_a:.3f}   (previously 0.906)
B (DDIM, n_steps=1000, w=3.0)       = {fid_b:.3f}   (previously 0.562)
E (DDIM, n_steps=250,  w=3.0, eta=0)= {fid_e:.3f}   (previously ~0.578)
F (DDIM, n_steps=250,  w=3.0, eta=1)= {fid_f:.3f}   (NEW -- should close most of the A-E gap)
C (DDIM, n_steps=250,  w=1.0)       = {fid_c:.3f}   (previously 0.203)
D (ancestral, full, w=1.0)          = {fid_d:.3f}   (previously 0.984)

If F is close to A (much closer than E was):
    -> confirms the remaining B/E gap under guidance was DDIM's eta=0
       determinism being fragile under CFG extrapolation, not a further
       bug. Re-run experiments.py with --ddim_eta somewhere in [0.3, 1.0]
       for Experiments 3/4, and report the eta/speed/robustness trade-off
       as a finding (fast+fragile vs. slower+robust).

If F is still far from A:
    -> there's something else going on beyond eta; send me these numbers.
""")


if __name__ == '__main__':
    main()