"""
Runnable, dependency-light (NumPy only) verification of every closed-form
formula used by the diffusion/VAE code, so correctness of the core math is
checked BEFORE it's used inside PyTorch training (in case PyTorch is not 
available in this environment).
Run: python verify/verify_math.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import numpy as np
from diffusion_utils import (
    linear_beta_schedule, cosine_beta_schedule, compute_alphas,
    q_sample, predict_z0_from_noise, kl_diag_gaussian_to_standard_normal,
    sinusoidal_time_embedding, ddpm_reverse_step_mean_var, ddim_step,
)

PASS = []
FAIL = []


def check(name, cond, detail=""):
    if cond:
        PASS.append(name)
        print(f"  [PASS] {name}")
    else:
        FAIL.append(name)
        print(f"  [FAIL] {name}  {detail}")


def test_beta_schedules():
    print("\n== beta schedules ==")
    T = 1000  # standard DDPM horizon (Ho et al. 2020) -- linear schedule is
              # calibrated (beta_start=1e-4, beta_end=0.02) for this length
    for name, betas in [("linear", linear_beta_schedule(T)),
                         ("cosine", cosine_beta_schedule(T))]:
        alphas, alpha_bars = compute_alphas(betas)
        check(f"{name}: betas in (0,1)", np.all(betas > 0) and np.all(betas < 1))
        check(f"{name}: alpha_bar monotonically decreasing",
              np.all(np.diff(alpha_bars) < 0))
        check(f"{name}: alpha_bar starts near 1, ends near 0",
              alpha_bars[0] > 0.9 and alpha_bars[-1] < 0.05,
              f"got start={alpha_bars[0]:.4f} end={alpha_bars[-1]:.6f}")
        check(f"{name}: alpha_bar in (0,1] everywhere",
              np.all(alpha_bars > 0) and np.all(alpha_bars <= 1.0))


def test_q_sample_statistics():
    print("\n== forward process q(z_t|z0) empirical statistics ==")
    rng = np.random.default_rng(0)
    T = 200
    betas = linear_beta_schedule(T)
    alphas, alpha_bars = compute_alphas(betas)

    D = 8
    z0 = rng.normal(0, 1, size=(1, D))  # a single fixed latent, broadcast
    n_samples = 20000
    for t_val in [0, 50, 100, 199]:
        t = np.full(n_samples, t_val)
        noise = rng.normal(0, 1, size=(n_samples, D))
        z0_rep = np.repeat(z0, n_samples, axis=0)
        z_t = q_sample(z0_rep, t, alpha_bars, noise)

        theoretical_mean = np.sqrt(alpha_bars[t_val]) * z0[0]
        theoretical_var = 1 - alpha_bars[t_val]

        empirical_mean = z_t.mean(axis=0)
        empirical_var = z_t.var(axis=0)

        mean_ok = np.allclose(empirical_mean, theoretical_mean, atol=0.05)
        var_ok = np.allclose(empirical_var, theoretical_var, atol=0.05)
        check(f"t={t_val}: empirical mean matches sqrt(alpha_bar_t)*z0", mean_ok,
              f"max abs diff={np.max(np.abs(empirical_mean - theoretical_mean)):.4f}")
        check(f"t={t_val}: empirical var matches (1-alpha_bar_t)", var_ok,
              f"max abs diff={np.max(np.abs(empirical_var - theoretical_var)):.4f}")

    # sanity: at t=0 with linear schedule beta_start=1e-4, z_t should be
    # almost identical to z0 (alpha_bar_0 = 1 - beta_0 ~ 0.9999)
    check("t=0 forward sample stays very close to z0 (low-noise limit)",
          alpha_bars[0] > 0.999)


def test_predict_z0_inversion():
    print("\n== predict_z0_from_noise correctly inverts q_sample ==")
    rng = np.random.default_rng(1)
    T = 100
    betas = linear_beta_schedule(T)
    alphas, alpha_bars = compute_alphas(betas)
    D = 4
    N = 500
    z0 = rng.normal(0, 1, size=(N, D))
    t = rng.integers(0, T, size=N)
    noise = rng.normal(0, 1, size=(N, D))
    z_t = q_sample(z0, t, alpha_bars, noise)
    # if we feed the model the TRUE noise, it should recover the true z0 exactly
    z0_recovered = predict_z0_from_noise(z_t, t, alpha_bars, noise)
    check("predict_z0_from_noise(q_sample(z0,t,noise), noise) == z0",
          np.allclose(z0_recovered, z0, atol=1e-8),
          f"max abs diff={np.max(np.abs(z0_recovered - z0)):.2e}")


def test_kl_divergence():
    print("\n== KL(N(mu,var) || N(0,I)) closed form vs. known special cases ==")
    # KL(N(0,1)||N(0,1)) == 0
    kl0 = kl_diag_gaussian_to_standard_normal(np.zeros((1, 5)), np.zeros((1, 5)))
    check("KL(N(0,I)||N(0,I)) == 0", np.allclose(kl0, 0.0), f"got {kl0}")

    # 1-D closed form check against the textbook formula for N(mu,sigma^2)||N(0,1):
    # KL = 0.5*(sigma^2 + mu^2 - 1 - log(sigma^2))
    rng = np.random.default_rng(2)
    mu = rng.normal(0, 2, size=(1000, 1))
    logvar = rng.normal(0, 1, size=(1000, 1))
    var = np.exp(logvar)
    textbook = 0.5 * (var + mu ** 2 - 1 - logvar)
    ours = kl_diag_gaussian_to_standard_normal(mu, logvar)[:, None]
    check("matches textbook scalar-Gaussian KL formula",
          np.allclose(textbook, ours, atol=1e-8),
          f"max abs diff={np.max(np.abs(textbook - ours)):.2e}")

    # KL is monotonically increasing in |mu| for fixed var
    mus = np.array([[0.0], [1.0], [2.0], [3.0]])
    logvars = np.zeros((4, 1))
    kls = kl_diag_gaussian_to_standard_normal(mus, logvars)
    check("KL increases with |mu - 0| for fixed variance",
          np.all(np.diff(kls) > 0), f"kls={kls}")

    # KL >= 0 always (Gibbs' inequality) over random draws
    mu = rng.normal(0, 3, size=(5000, 4))
    logvar = rng.normal(0, 2, size=(5000, 4))
    kl = kl_diag_gaussian_to_standard_normal(mu, logvar)
    check("KL >= 0 for all random (mu, logvar) draws", np.all(kl >= -1e-10),
          f"min={kl.min():.2e}")


def test_reverse_step_matches_true_posterior():
    """Ho et al. 2020 (eq. 11-12): if the model's predicted noise equals the
    TRUE noise used to generate z_t from z0, then mu_theta(z_t,t) computed
    via ddpm_reverse_step_mean_var must equal the exact posterior mean
    mu_tilde(z_t, z0, t) of q(z_{t-1} | z_t, z0). This is the key identity
    the whole DDPM reverse-process parameterization relies on -- if this
    fails, sampling is silently wrong even though training might look fine."""
    print("\n== reverse (denoising) step matches true posterior mean ==")
    rng = np.random.default_rng(3)
    T = 200
    betas = linear_beta_schedule(T)
    alphas, alpha_bars = compute_alphas(betas)
    alpha_bars_prev = np.concatenate([[1.0], alpha_bars[:-1]])

    D, N = 6, 300
    z0 = rng.normal(0, 1, size=(N, D))
    t = rng.integers(1, T, size=N)  # avoid t=0 edge case (alpha_bar_prev=1 trivial)
    noise = rng.normal(0, 1, size=(N, D))
    z_t = q_sample(z0, t, alpha_bars, noise)

    mean_model, var_model = ddpm_reverse_step_mean_var(z_t, t, noise, betas, alphas, alpha_bars)

    ab_t = alpha_bars[t][:, None]
    ab_prev = alpha_bars_prev[t][:, None]
    a_t = alphas[t][:, None]
    b_t = betas[t][:, None]
    mu_tilde = (np.sqrt(ab_prev) * b_t / (1 - ab_t)) * z0 + \
               (np.sqrt(a_t) * (1 - ab_prev) / (1 - ab_t)) * z_t
    posterior_var_true = b_t[:, 0] * (1 - ab_prev[:, 0]) / (1 - ab_t[:, 0])

    check("mu_theta(z_t,t; true noise) == true posterior mean mu_tilde",
          np.allclose(mean_model, mu_tilde, atol=1e-6),
          f"max abs diff={np.max(np.abs(mean_model - mu_tilde)):.2e}")
    check("model's fixed variance (beta_t) is a valid upper bound on the "
          "true posterior variance (Ho et al. Sec 3.2)",
          np.all(var_model >= posterior_var_true - 1e-8))


def test_ddim_z0_amplification_requires_clipping():
    """Regression test for a real bug found when this project was first run:
    DDIM's z0_pred = (z_t - sqrt(1-abar_t)*eps_pred) / sqrt(abar_t) divides
    by a near-zero number as t -> T (abar_t -> 0), amplifying any small
    denoiser prediction error by orders of magnitude and silently wrecking
    sample quality -- even with NO step-skipping (n_steps == T) and NO
    guidance. This documents the failure mode numerically and confirms
    that clipping z0_pred to a data-driven range (as diffusion_model.py's
    LatentDiffusion now does) bounds the damage. See diffusion_model.py's
    LatentDiffusion docstring / __init__ comment for the fix."""
    print("\n== DDIM z0-prediction amplification at high t (regression test) ==")
    rng = np.random.default_rng(7)
    T = 1000
    betas = cosine_beta_schedule(T)
    alphas, alpha_bars = compute_alphas(betas)
    D = 16
    z0_true = rng.normal(0, 1, size=D)
    error_std = 0.15  # realistic small denoiser prediction error

    def z0_pred_at(t, clip=None):
        noise_true = rng.normal(0, 1, size=D)
        z_t = q_sample(z0_true[None, :], np.array([t]), alpha_bars, noise_true[None, :])[0]
        eps_pred = noise_true + rng.normal(0, error_std, size=D)
        z0p = (z_t - np.sqrt(1 - alpha_bars[t]) * eps_pred) / np.sqrt(alpha_bars[t])
        if clip is not None:
            z0p = np.clip(z0p, -clip, clip)
        return z0p

    err_t0_unclipped = np.linalg.norm(z0_pred_at(0) - z0_true)
    err_t999_unclipped = np.linalg.norm(z0_pred_at(999) - z0_true)
    check("unclipped z0_pred error explodes at t=T-1 vs t=0 (documents the bug)",
          err_t999_unclipped > 100 * err_t0_unclipped,
          f"t=0 err={err_t0_unclipped:.3f}, t=999 err={err_t999_unclipped:.3f}")

    clip_bound = 4.0  # e.g. 4 std under a roughly unit-variance latent prior
    err_t999_clipped = np.linalg.norm(z0_pred_at(999, clip=clip_bound) - z0_true)
    check("clipping z0_pred bounds the t=999 error to a sane range (the fix works)",
          err_t999_clipped < 50,
          f"clipped t=999 err={err_t999_clipped:.3f}")


def test_ddim_eta1_matches_ancestral():
    """Known identity (Song et al. 2020, DDIM paper, Sec 4.1): with eta=1
    and consecutive (non-skipped) timesteps, generalized DDIM is
    mathematically identical to DDPM ancestral sampling -- both the mean
    AND the variance. This is what justifies adding an `eta` parameter to
    sample_ddim as a genuine speed/robustness dial rather than an
    unrelated new mechanism: eta=0 is fast+fragile-under-guidance,
    eta=1 recovers ancestral's robustness."""
    print("\n== DDIM(eta=1, consecutive steps) == DDPM ancestral (known identity) ==")
    rng = np.random.default_rng(9)
    T = 300
    betas = linear_beta_schedule(T)
    alphas, alpha_bars = compute_alphas(betas)

    D, N = 5, 200
    z_t = rng.normal(0, 1, size=(N, D))
    eps_pred = rng.normal(0, 1, size=(N, D))
    t = rng.integers(1, T, size=N)

    # DDPM ancestral reverse step (already-verified formula)
    ddpm_mean, ddpm_var = ddpm_reverse_step_mean_var(z_t, t, eps_pred, betas, alphas, alpha_bars)

    # generalized DDIM step with eta=1, t_prev = t-1 (no skipping), per-sample
    # DDPM ancestral TRUE posterior variance beta_tilde_t = (1-abar_{t-1})/(1-abar_t)*beta_t
    # -- NOT the "simple" var=beta_t choice returned by ddpm_reverse_step_mean_var
    # (that's a different, also-valid, upper-bound choice from Ho et al. 2020
    # Sec 3.2; the eta=1 DDIM identity specifically matches the TRUE posterior
    # variance, not the simple upper bound -- see Song et al. 2020 Sec 4.1).
    alpha_bars_prev_arr = np.concatenate([[1.0], alpha_bars[:-1]])
    beta_tilde = (1 - alpha_bars_prev_arr[t]) / (1 - alpha_bars[t]) * betas[t]

    ddim_means = np.zeros_like(z_t)
    ddim_sigmas = np.zeros(N)
    for i in range(N):
        m, s, _ = ddim_step(z_t[i], t[i], t[i] - 1, eps_pred[i], alpha_bars, eta=1.0)
        ddim_means[i] = m
        ddim_sigmas[i] = s

    check("eta=1 DDIM mean matches DDPM ancestral mean",
          np.allclose(ddim_means, ddpm_mean, atol=1e-5),
          f"max abs diff={np.max(np.abs(ddim_means - ddpm_mean)):.2e}")
    check("eta=1 DDIM variance (sigma^2) matches the TRUE DDPM posterior "
          "variance beta_tilde_t (not the simple beta_t upper bound)",
          np.allclose(ddim_sigmas ** 2, beta_tilde, atol=1e-8),
          f"max abs diff={np.max(np.abs(ddim_sigmas**2 - beta_tilde)):.2e}")
    check("sanity: the simple beta_t choice really is an upper bound on beta_tilde_t "
          "(confirms these are legitimately two different quantities, not a bug)",
          np.all(ddpm_var >= beta_tilde - 1e-10))

    # eta=0 must always be perfectly deterministic (sigma == 0)
    m0, s0, _ = ddim_step(z_t[0], t[0], t[0] - 1, eps_pred[0], alpha_bars, eta=0.0)
    check("eta=0 gives exactly zero stochasticity (sigma=0)", s0 == 0.0, f"got sigma={s0}")

    # sigma should scale ~linearly with eta in between (sanity, not a deep identity)
    _, s_half, _ = ddim_step(z_t[0], t[0], t[0] - 1, eps_pred[0], alpha_bars, eta=0.5)
    check("sigma(eta=0.5) ~= 0.5 * sigma(eta=1.0) (linear in eta, per the formula)",
          np.isclose(s_half, 0.5 * ddim_sigmas[0], atol=1e-6),
          f"s_half={s_half:.4f}, 0.5*s1={0.5*ddim_sigmas[0]:.4f}")


def test_time_embedding():
    print("\n== sinusoidal time embedding ==")
    emb = sinusoidal_time_embedding(np.array([0, 1, 10, 100]), dim=16)
    check("shape is (4, 16)", emb.shape == (4, 16))
    check("values bounded in [-1, 1] (sin/cos outputs)",
          np.all(emb >= -1.0001) and np.all(emb <= 1.0001))
    check("different timesteps give different embeddings",
          not np.allclose(emb[0], emb[1]))
    emb0 = sinusoidal_time_embedding(np.array([5]), dim=16)
    emb0b = sinusoidal_time_embedding(np.array([5]), dim=16)
    check("embedding is deterministic (same t -> same vector)",
          np.allclose(emb0, emb0b))


if __name__ == '__main__':
    test_beta_schedules()
    test_q_sample_statistics()
    test_predict_z0_inversion()
    test_reverse_step_matches_true_posterior()
    test_ddim_z0_amplification_requires_clipping()
    test_ddim_eta1_matches_ancestral()
    test_kl_divergence()
    test_time_embedding()

    print(f"\n{'='*50}\n{len(PASS)} passed, {len(FAIL)} failed\n{'='*50}")
    if FAIL:
        print("FAILED:", FAIL)
        sys.exit(1)
    else:
        print("All diffusion/VAE math verified correct (NumPy, no PyTorch needed).")