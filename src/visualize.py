import sys, os, glob, re
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
FIG_DIR = os.path.join(os.path.dirname(__file__), '..', 'figures')
os.makedirs(FIG_DIR, exist_ok=True)

plt.rcParams.update({
    'figure.dpi': 130, 'savefig.dpi': 130, 'font.size': 11,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.grid': True, 'grid.alpha': 0.25,
})


def _imshow_grid(ax, images, nrow=4, title=''):
    """images: [N, 1, H, W] in [-1,1] -> tiled grayscale grid."""
    images = np.clip((images + 1) / 2, 0, 1)  # back to [0,1]
    n, _, h, w = images.shape
    ncol = int(np.ceil(n / nrow))
    canvas = np.ones((nrow * h, ncol * w))
    for i in range(n):
        r, c = i // ncol, i % ncol
        canvas[r * h:(r + 1) * h, c * w:(c + 1) * w] = images[i, 0]
    ax.imshow(canvas, cmap='gray', vmin=0, vmax=1)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, fontsize=9)


def fig_latent_dim_study():
    df = pd.read_csv(os.path.join(RESULTS_DIR, 'exp1_latent_dim_study.csv'))
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))

    axes[0].plot(df.latent_dim, df.test_recon, marker='o', color='#4C72B0')
    axes[0].set_xlabel('Latent dimension'); axes[0].set_ylabel('Test reconstruction loss')
    axes[0].set_title('Reconstruction fidelity vs. latent dim')
    axes[0].set_xscale('log', base=2)

    axes[1].plot(df.latent_dim, df.train_kl, marker='o', color='#C44E52')
    axes[1].set_xlabel('Latent dimension'); axes[1].set_ylabel('KL divergence (nats)')
    axes[1].set_title('Posterior KL vs. latent dim')
    axes[1].set_xscale('log', base=2)

    axes[2].plot(df.latent_dim, df.prior_sample_diversity, marker='o', color='#55A868')
    axes[2].set_xlabel('Latent dimension'); axes[2].set_ylabel('Prior-sample diversity (mean pairwise L2)')
    axes[2].set_title('Unconditional sample diversity vs. latent dim')
    axes[2].set_xscale('log', base=2)

    fig.suptitle('Experiment 1 — Effect of latent-space dimensionality', y=1.03, fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'exp1_latent_dim_study.png'), bbox_inches='tight')
    plt.close(fig)

    # latent-space structure scatter plots (2D / PCA-projected)
    proj_files = sorted(glob.glob(os.path.join(RESULTS_DIR, 'exp1_latent_proj_d*.npz')),
                         key=lambda f: int(re.search(r'_d(\d+)\.npz', f).group(1)))
    if proj_files:
        n = len(proj_files)
        fig, axes = plt.subplots(1, n, figsize=(3.3 * n, 3.3))
        axes = np.atleast_1d(axes)
        for ax, f in zip(axes, proj_files):
            d = int(re.search(r'_d(\d+)\.npz', f).group(1))
            data = np.load(f)
            sc = ax.scatter(data['proj'][:, 0], data['proj'][:, 1], c=data['labels'],
                             cmap='tab10', s=4, alpha=0.6)
            ax.set_title(f'dim={d}' + (' (raw 2D)' if d == 2 else ' (PCA→2D)'), fontsize=9)
            ax.set_xticks([]); ax.set_yticks([])
        fig.suptitle('Latent space structure by class label, across latent dims', y=1.05)
        fig.colorbar(sc, ax=axes.tolist(), label='class', shrink=0.8)
        fig.savefig(os.path.join(FIG_DIR, 'exp1_latent_structure.png'), bbox_inches='tight')
        plt.close(fig)

    # unconditional prior samples, across dims
    sample_files = sorted(glob.glob(os.path.join(RESULTS_DIR, 'exp1_prior_samples_d*.npy')),
                           key=lambda f: int(re.search(r'_d(\d+)\.npy', f).group(1)))
    if sample_files:
        n = len(sample_files)
        fig, axes = plt.subplots(1, n, figsize=(2.6 * n, 3))
        axes = np.atleast_1d(axes)
        for ax, f in zip(axes, sample_files):
            d = int(re.search(r'_d(\d+)\.npy', f).group(1))
            imgs = np.load(f)
            _imshow_grid(ax, imgs, nrow=4, title=f'dim={d}')
        fig.suptitle('Unconditional samples from the prior N(0,I), across latent dims', y=1.03)
        fig.savefig(os.path.join(FIG_DIR, 'exp1_prior_samples.png'), bbox_inches='tight')
        plt.close(fig)


def fig_conditional_generation():
    df = pd.read_csv(os.path.join(RESULTS_DIR, 'exp2_conditional_generation.csv'))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    axes[0].bar(df.class_name, df.fidelity, color='#4C72B0')
    axes[0].set_ylabel('Classifier fidelity (target class match rate)')
    axes[0].set_title('Experiment 2 — Conditional generation fidelity per class')
    axes[0].tick_params(axis='x', rotation=45)
    axes[0].axhline(df.fidelity.mean(), color='black', linestyle='--', linewidth=1,
                     label=f'mean={df.fidelity.mean():.2f}')
    axes[0].legend(fontsize=8)

    axes[1].bar(df.class_name, df.diversity, color='#DD8452')
    axes[1].set_ylabel('Within-class sample diversity (mean pairwise L2)')
    axes[1].set_title('Sample diversity per class')
    axes[1].tick_params(axis='x', rotation=45)

    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'exp2_conditional_generation.png'), bbox_inches='tight')
    plt.close(fig)

    grid_path = os.path.join(RESULTS_DIR, 'exp2_example_grids.npz')
    if os.path.exists(grid_path):
        data = np.load(grid_path)
        keys = sorted(data.files, key=lambda k: int(k.split('_')[1]))
        n = len(keys)
        fig, axes = plt.subplots(1, n, figsize=(2.0 * n, 2.6))
        axes = np.atleast_1d(axes)
        for ax, k in zip(axes, keys):
            _imshow_grid(ax, data[k], nrow=4, title=k.replace('class_', 'class '))
        fig.suptitle('Example conditionally-generated samples per class', y=1.05)
        fig.savefig(os.path.join(FIG_DIR, 'exp2_example_samples.png'), bbox_inches='tight')
        plt.close(fig)


def fig_ddim_steps():
    df = pd.read_csv(os.path.join(RESULTS_DIR, 'exp3_ddim_steps.csv'))
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.plot(df.n_steps, df.fidelity, marker='o', color='#4C72B0', label='Fidelity')
    ax.set_xlabel('Number of DDIM sampling steps')
    ax.set_ylabel('Classifier fidelity', color='#4C72B0')
    ax.set_xscale('log')
    ax2 = ax.twinx()
    ax2.plot(df.n_steps, df.wall_time_s, marker='s', linestyle='--', color='gray', label='Wall time')
    ax2.set_ylabel('Sampling wall time (s)', color='gray')
    ax.set_title('Experiment 3 — Sampling speed vs. quality (DDIM)')
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'exp3_ddim_steps.png'), bbox_inches='tight')
    plt.close(fig)


def fig_guidance_scale():
    df = pd.read_csv(os.path.join(RESULTS_DIR, 'exp4_guidance_scale.csv'))
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.plot(df.guidance_scale, df.mean_fidelity, marker='o', color='#4C72B0', label='Fidelity')
    ax.set_xlabel('Classifier-free guidance scale (w)')
    ax.set_ylabel('Mean fidelity', color='#4C72B0')
    ax2 = ax.twinx()
    ax2.plot(df.guidance_scale, df.mean_diversity, marker='s', linestyle='--',
              color='gray', label='Diversity')
    ax2.set_ylabel('Mean within-class diversity', color='gray')
    ax.set_title('Experiment 4 — Guidance scale: fidelity/diversity trade-off')
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'exp4_guidance_scale.png'), bbox_inches='tight')
    plt.close(fig)


def fig_diffusion_training_curve():
    path = os.path.join(RESULTS_DIR, 'diffusion_training_curve.csv')
    if not os.path.exists(path):
        return
    df = pd.read_csv(path)
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.plot(df.epoch, df.loss, color='#4C72B0')
    ax.set_xlabel('Epoch'); ax.set_ylabel('Denoising MSE loss')
    ax.set_title('Conditional latent-diffusion training curve')
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'diffusion_training_curve.png'), bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    fig_latent_dim_study()
    fig_conditional_generation()
    fig_ddim_steps()
    fig_guidance_scale()
    fig_diffusion_training_curve()
    print("All figures written to", FIG_DIR)
