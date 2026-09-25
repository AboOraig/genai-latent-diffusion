"""
Runs the controlled Experiment 4b (eta x guidance_scale sweep on a SINGLE
checkpoint) using your already-trained VAE + diffusion model, without
retraining anything. Much faster than re-running experiments.py just to
get this comparison.

Usage (from src/, after you've run experiments.py at least once so
../results/vae_d16.pt and ../results/denoiser.pt exist):

    python run_eta_sweep.py

Then:
    python visualize.py        # adds figures/exp4b_eta_sweep.png
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import torch

from vae import VAE
from diffusion_model import Denoiser, LatentDiffusion
from classifier import train_classifier
from train_utils import encode_dataset
from data import get_dataloaders
from experiments import experiment_4b_eta_sweep

LATENT_DIM = 16
NUM_CLASSES = 10
T = 1000
SCHEDULE = 'cosine'


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

    print("(re-encoding training set to get latent stats for DDIM clipping)")
    z_train, _ = encode_dataset(vae, train_loader, device)
    latent_mean, latent_std = z_train.mean(axis=0), z_train.std(axis=0)

    diffusion = LatentDiffusion(denoiser, NUM_CLASSES, T=T, schedule=SCHEDULE, device=device,
                                 latent_mean=latent_mean, latent_std=latent_std)

    print("(re-training a quick classifier for scoring -- ~1 min)")
    classifier = train_classifier(train_loader, test_loader, device, epochs=2)

    print("\n=== Experiment 4b: controlled eta ablation (single checkpoint) ===")
    df = experiment_4b_eta_sweep(
        vae, diffusion, classifier, device, LATENT_DIM,
        eta_values=[0.0, 0.25, 0.5, 0.75, 1.0],
        guidance_scales=[1.0, 4.0, 7.0, 10.0],
    )
    print("\nSaved to ../results/exp4b_eta_sweep.csv")
    print(df.pivot(index='eta', columns='guidance_scale', values='mean_fidelity'))


if __name__ == '__main__':
    main()
