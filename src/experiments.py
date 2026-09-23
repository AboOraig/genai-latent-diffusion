"""
Runs the four computational studies:
  1. Latent-space dimensionality study (VAE)
  2. Conditional latent diffusion: per-class generation quality & diversity
  3. Sampling-steps (DDIM) ablation: speed vs. quality
  4. Classifier-free-guidance-scale ablation: fidelity vs. diversity

Usage:
    python experiments.py --dataset mnist --epochs_vae 15 --epochs_diffusion 100

All results are written to ../results/*.csv and ../results/*.npz (raw
arrays for figures); run visualize.py afterwards to produce ../figures/*.png
and generate_report.py to populate ../REPORT.md with the real numbers.
"""
import sys, os, time, argparse
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import pandas as pd
import torch

from data import get_dataloaders
from train_utils import train_vae, encode_dataset, train_latent_diffusion
from classifier import train_classifier, classification_fidelity
from vae import VAE

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)


def pairwise_diversity(images):
    """Mean pairwise L2 distance between flattened images -- a simple,
    dependency-free diversity proxy (no FID/Inception features needed)."""
    x = images.reshape(images.shape[0], -1)
    if x.shape[0] < 2:
        return 0.0
    d = np.linalg.norm(x[:, None, :] - x[None, :, :], axis=-1)
    iu = np.triu_indices(x.shape[0], k=1)
    return float(d[iu].mean())


def experiment_1_latent_dim(train_loader, test_loader, device, latent_dims, epochs):
    rows = []
    saved = {}
    for d in latent_dims:
        print(f"\n[exp1] training VAE with latent_dim={d}")
        model, history = train_vae(train_loader, test_loader, d, device, epochs=epochs)
        final = history[-1]

        # unconditional prior samples: quality/diversity as a function of dim
        model.eval()
        with torch.no_grad():
            z = torch.randn(64, d, device=device)
            samples = model.decoder(z).cpu().numpy()
        diversity = pairwise_diversity(samples)

        # latent structure: encode test set, project to 2D for visualization
        z_test, y_test = encode_dataset(model, test_loader, device)
        if d > 2:
            from sklearn.decomposition import PCA
            proj = PCA(n_components=2).fit_transform(z_test[:2000])
        else:
            proj = z_test[:2000]
        np.savez(os.path.join(RESULTS_DIR, f'exp1_latent_proj_d{d}.npz'),
                 proj=proj, labels=y_test[:2000])
        np.save(os.path.join(RESULTS_DIR, f'exp1_prior_samples_d{d}.npy'), samples[:16])

        rows.append(dict(latent_dim=d, train_recon=final['train_recon'],
                          train_kl=final['train_kl'], test_recon=final['test_recon'],
                          prior_sample_diversity=diversity))
        saved[d] = model

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS_DIR, 'exp1_latent_dim_study.csv'), index=False)
    print("\n[exp1] done:\n", df)
    return df, saved


def experiment_2_conditional_generation(vae, diffusion, classifier, class_names, device,
                                         latent_dim, n_per_class=32, guidance_scale=3.0):
    num_classes = len(class_names)
    rows = []
    example_grids = {}
    for c in range(num_classes):
        y = torch.full((n_per_class,), c, dtype=torch.long, device=device)
        z_gen = diffusion.sample(n_per_class, y, guidance_scale=guidance_scale, latent_dim=latent_dim)
        with torch.no_grad():
            imgs = vae.decoder(z_gen)
        fidelity = classification_fidelity(classifier, imgs, y, device)
        diversity = pairwise_diversity(imgs.cpu().numpy())
        rows.append(dict(class_idx=c, class_name=class_names[c],
                          fidelity=fidelity, diversity=diversity))
        example_grids[c] = imgs[:8].cpu().numpy()
        print(f"  [exp2] class={class_names[c]:>12s}  fidelity={fidelity:.3f}  diversity={diversity:.2f}")

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS_DIR, 'exp2_conditional_generation.csv'), index=False)
    np.savez(os.path.join(RESULTS_DIR, 'exp2_example_grids.npz'),
              **{f'class_{c}': g for c, g in example_grids.items()})
    return df


def experiment_3_ddim_steps(vae, diffusion, classifier, device, latent_dim,
                             step_counts, n_samples=64, fixed_class=0, guidance_scale=3.0):
    rows = []
    for n_steps in step_counts:
        y = torch.full((n_samples,), fixed_class, dtype=torch.long, device=device)
        t0 = time.time()
        z_gen = diffusion.sample_ddim(n_samples, y, guidance_scale=guidance_scale,
                                       latent_dim=latent_dim, n_steps=n_steps)
        elapsed = time.time() - t0
        with torch.no_grad():
            imgs = vae.decoder(z_gen)
        fidelity = classification_fidelity(classifier, imgs, y, device)
        rows.append(dict(n_steps=n_steps, wall_time_s=elapsed, fidelity=fidelity))
        print(f"  [exp3] n_steps={n_steps:4d}  time={elapsed:.3f}s  fidelity={fidelity:.3f}")
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS_DIR, 'exp3_ddim_steps.csv'), index=False)
    return df


def experiment_4_guidance_scale(vae, diffusion, classifier, device, latent_dim,
                                 guidance_scales, n_per_class=32, n_steps=50, n_classes_eval=10):
    rows = []
    for w in guidance_scales:
        fids, divs = [], []
        for c in range(n_classes_eval):
            y = torch.full((n_per_class,), c, dtype=torch.long, device=device)
            z_gen = diffusion.sample_ddim(n_per_class, y, guidance_scale=w,
                                           latent_dim=latent_dim, n_steps=n_steps)
            with torch.no_grad():
                imgs = vae.decoder(z_gen)
            fids.append(classification_fidelity(classifier, imgs, y, device))
            divs.append(pairwise_diversity(imgs.cpu().numpy()))
        rows.append(dict(guidance_scale=w, mean_fidelity=np.mean(fids), mean_diversity=np.mean(divs)))
        print(f"  [exp4] w={w:5.1f}  mean_fidelity={np.mean(fids):.3f}  mean_diversity={np.mean(divs):.2f}")
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS_DIR, 'exp4_guidance_scale.csv'), index=False)
    return df


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', default='mnist', choices=['mnist', 'fashion_mnist'])
    p.add_argument('--batch_size', type=int, default=128)
    p.add_argument('--epochs_vae', type=int, default=15)
    p.add_argument('--epochs_diffusion', type=int, default=100)
    p.add_argument('--epochs_classifier', type=int, default=3)
    p.add_argument('--main_latent_dim', type=int, default=16,
                    help='latent dim used for Experiments 2-4 (diffusion, guidance)')
    p.add_argument('--latent_dims', type=int, nargs='+', default=[2, 4, 8, 16, 32, 64])
    p.add_argument('--ddim_steps', type=int, nargs='+', default=[5, 10, 20, 50, 100, 250])
    p.add_argument('--guidance_scales', type=float, nargs='+', default=[0.0, 1.0, 2.0, 4.0, 7.0, 10.0])
    args = p.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("device:", device)

    train_loader, test_loader, class_names = get_dataloaders(args.dataset, args.batch_size)

    print("\n=== training evaluation classifier ===")
    classifier = train_classifier(train_loader, test_loader, device, epochs=args.epochs_classifier)

    print("\n=== Experiment 1: latent dimensionality study ===")
    exp1_df, vae_models = experiment_1_latent_dim(
        train_loader, test_loader, device, args.latent_dims, args.epochs_vae)

    if args.main_latent_dim in vae_models:
        vae = vae_models[args.main_latent_dim]
    else:
        print(f"\n[main] main_latent_dim={args.main_latent_dim} not in --latent_dims, training it separately")
        vae, _ = train_vae(train_loader, test_loader, args.main_latent_dim, device, epochs=args.epochs_vae)

    print(f"\n=== encoding dataset into latent space (dim={args.main_latent_dim}) ===")
    z_train, y_train = encode_dataset(vae, train_loader, device)

    print("\n=== training conditional latent diffusion ===")
    diffusion, diff_history = train_latent_diffusion(
        z_train, y_train, num_classes=len(class_names), device=device,
        latent_dim=args.main_latent_dim, epochs=args.epochs_diffusion)
    pd.DataFrame(diff_history).to_csv(
        os.path.join(RESULTS_DIR, 'diffusion_training_curve.csv'), index=False)

    print("\n=== Experiment 2: conditional generation quality/diversity per class ===")
    experiment_2_conditional_generation(vae, diffusion, classifier, class_names, device,
                                         args.main_latent_dim)

    print("\n=== Experiment 3: DDIM sampling-steps ablation ===")
    experiment_3_ddim_steps(vae, diffusion, classifier, device, args.main_latent_dim,
                             args.ddim_steps)

    print("\n=== Experiment 4: classifier-free-guidance-scale ablation ===")
    experiment_4_guidance_scale(vae, diffusion, classifier, device, args.main_latent_dim,
                                 args.guidance_scales)

    torch.save(vae.state_dict(), os.path.join(RESULTS_DIR, f'vae_d{args.main_latent_dim}.pt'))
    torch.save(diffusion.denoiser.state_dict(), os.path.join(RESULTS_DIR, 'denoiser.pt'))
    print("\nAll experiments complete. Run visualize.py next, then generate_report.py.")


if __name__ == '__main__':
    main()
