import os
import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize
from scipy.integrate import trapezoid
import matplotlib.pyplot as plt
from itertools import combinations

RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)


def savefig(name):
    plt.savefig(os.path.join(RESULTS_DIR, name), dpi=150, bbox_inches='tight')
    plt.close()


# ── Stimulus grid ──────────────────────────────────────────────────────────────

n_stim = 11
S      = np.linspace(0, 1, n_stim, endpoint=False)
S_grid = np.linspace(0, 1, 1000, endpoint=False)


# ── Parametric model (Prior = ψ'(s) by construction) ──────────────────────────
#
#   ψ(s; A, μ)  = s + A/(2π) · sin(2π(s − μ))
#   ψ'(s; A, μ) = 1 + A·cos(2π(s − μ))          ← this IS the prior
#
#   |A| < 1  ⟹  ψ'(s) > 0 everywhere (monotone ψ, valid prior, integrates to 1).
#
#     bias = inv_J · (log p)' + (inv_J)'
#          = 2πAσ² · sin(2π(s−μ)) / ψ'(s)³

def psi(s, A, mu):
    return s + (A / (2 * np.pi)) * np.sin(2 * np.pi * (s - mu))

def dpsi(s, A, mu):
    """ψ'(s; A, μ) — also equals the prior."""
    return 1 + A * np.cos(2 * np.pi * (s - mu))

def prior_fn(s, A, mu):
    """Prior = ψ'(s) by construction."""
    return dpsi(s, A, mu)

def analytical_bias(s, A, mu, sigma):
    return 2 * np.pi * A * sigma**2 * np.sin(2 * np.pi * (s - mu)) / dpsi(s, A, mu)**3

def expected_percept(s, A, mu, sigma):
    return s + analytical_bias(s, A, mu, sigma)

def d2log_prior_fn(s, A, mu):
    """(log prior)''(s) = −4π²A·(cos(2π(s−μ)) + A) / (1+A·cos(2π(s−μ)))²"""
    c = np.cos(2 * np.pi * (s - mu))
    return -4 * np.pi**2 * A * (c + A) / (1 + A * c)**2

def percept_var(s, A, mu, sigma):
    """Exact Var[E[s|r]] = σ²ψ'² / (ψ'² − σ²·(log p)'')²
    Corrects σ²/ψ'² for prior curvature; the trough correction reaches ~20%
    at σ=0.05, causing A to be underestimated without this fix.
    """
    psi_p  = dpsi(s, A, mu)
    d2logp = d2log_prior_fn(s, A, mu)
    return sigma**2 * psi_p**2 / (psi_p**2 - sigma**2 * d2logp)**2

def rescale(x):
    return (x - x.min()) / (x.max() - x.min())


# ── True parameters ────────────────────────────────────────────────────────────

A_true     = 0.3    # amplitude  (|A| < 1 required)
mu_true    = 0.15   # prior peak location
sigma_true = 0.02   # noise — must be large enough relative to stimulus spacing (1/n_stim)
                    # so that enough triplets are non-saturated and carry info about A, μ.
                    # With σ=0.02 and spacing=1/11≈0.09, ratio≈4.5: almost all triplets
                    # saturate (P≈0 or 1) and A,μ are unidentifiable.
                    # With σ=0.05, ratio≈1.8: ~100/165 triplets are informative.

psi_vals = rescale(psi(S, A_true, mu_true))   # [0,1]-normalised for MLDS anchor constraints


# ── Part 1: Standard MLDS (Maloney & Yang 2003) ────────────────────────────────
#
# Triplet task: given ordered stimuli i < j < k, the observer decides whether
# |ψᵢ − ψⱼ| > |ψⱼ − ψₖ|.
# Decision variable: D = |ψᵢ − ψⱼ| − |ψⱼ − ψₖ| + ε,  ε ~ N(0, σ²).

plt.plot(S, psi_vals, 'k-o')
plt.xlabel('Stimulus s')
plt.ylabel('ψ(s)')
plt.title('True perceptual scale')
plt.grid(True, alpha=0.3)
savefig('01_true_perceptual_scale.png')


def simulate_trial(i, j, k):
    d1 = abs(psi_vals[i] - psi_vals[j])
    d2 = abs(psi_vals[j] - psi_vals[k])
    return int((d1 - d2) + np.random.normal(0, sigma_true) > 0)


n_repeats = 5
triplets  = list(combinations(range(n_stim), 3))   # C(11,3) = 165 unique triplets
data      = []

for _ in range(n_repeats):
    for i, j, k in triplets:
        data.append([i, j, k, simulate_trial(i, j, k)])

data = np.array(data)
print(f"{len(triplets)} triplets × {n_repeats} repeats = {len(data)} trials")


def reconstruct_psi(params):
    psi_ = np.zeros(n_stim)
    psi_[0]    = 0
    psi_[-1]   = 1
    psi_[1:-1] = params
    return psi_


def neg_log_likelihood_mlds(params, data, sigma):
    psi_  = reconstruct_psi(params)
    ii    = data[:, 0].astype(int)
    jj    = data[:, 1].astype(int)
    kk    = data[:, 2].astype(int)
    choices = data[:, 3]
    d1 = np.abs(psi_[ii] - psi_[jj])
    d2 = np.abs(psi_[jj] - psi_[kk])
    p  = np.clip(norm.cdf((d1 - d2) / sigma), 1e-10, 1 - 1e-10)
    return -(choices * np.log(p) + (1 - choices) * np.log(1 - p)).sum()


result  = minimize(neg_log_likelihood_mlds, np.linspace(0, 1, n_stim)[1:-1],
                   args=(data, sigma_true), method='L-BFGS-B')
psi_hat = reconstruct_psi(result.x)

plt.plot(S, psi_vals, 'k-o', label='True ψ')
plt.plot(S, psi_hat, 'g--s', label='MLDS recovered')
plt.xlabel('Stimulus s')
plt.ylabel('ψ(s)')
plt.legend()
plt.grid(True, alpha=0.3)
savefig('02_mlds_recovery.png')


# ── Part 2: Bayesian observer — posterior mean vs analytical bias ───────────────
#
# Observer receives r = ψ(s) + ε and computes E[s | r].
# Bias formula:
#   E[ŝ | s] − s = (1/J)·(log f_S)' + (1/J)'  + o(σ²)
# where J(s) = ψ'(s)² / σ² is Fisher information.

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].plot(S_grid, psi(S_grid, A_true, mu_true), 'k-', lw=2)
axes[0].plot(S_grid, S_grid, 'k--', lw=1, alpha=0.4, label='Identity')
axes[0].set_xlabel('s'); axes[0].set_ylabel('ψ(s)')
axes[0].set_title(f'Encoding function  (A={A_true}, μ={mu_true})')
axes[0].grid(True, alpha=0.3)

axes[1].plot(S_grid, prior_fn(S_grid, A_true, mu_true), 'b-', lw=2)
axes[1].set_xlabel('s'); axes[1].set_ylabel("p(s) = ψ'(s)")
axes[1].set_title("Prior = ψ'(s)")
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
savefig('03_encoding_and_prior.png')


def _posterior(s, A, mu, sigma, n_obs=1):
    """Normalised posterior on S_grid given n_obs noisy measurements of s."""
    r_samples = psi(s, A, mu) + np.random.normal(0, sigma, n_obs)
    log_lik   = np.sum(
        norm.logpdf(r_samples[:, None], psi(S_grid, A, mu)[None, :], sigma),
        axis=0,
    )
    log_lik  -= log_lik.max()
    posterior = np.exp(log_lik) * prior_fn(S_grid, A, mu)
    posterior /= trapezoid(posterior, S_grid)
    return posterior


def bayesian_percept_linear(s, A, mu, sigma, n_obs=1):
    """Posterior mean under MSE loss: E[s | r] = ∫ s · p(s|r) ds."""
    return trapezoid(S_grid * _posterior(s, A, mu, sigma, n_obs), S_grid)


def bayesian_percept_circular(s, A, mu, sigma, n_obs=1):
    """Posterior mean under circular loss: angle of E[e^{2πis} | r], mapped to [0,1)."""
    z = trapezoid(np.exp(2j * np.pi * S_grid) * _posterior(s, A, mu, sigma, n_obs), S_grid)
    return np.angle(z) / (2 * np.pi) % 1


# MC estimates: linear and circular posterior mean
n_mc              = 10000
expected_linear   = np.zeros(n_stim)
expected_circular = np.zeros(n_stim)

for idx, s in enumerate(S):
    lin_samples  = np.array([bayesian_percept_linear(s, A_true, mu_true, sigma_true)   for _ in range(n_mc)])
    circ_samples = np.array([bayesian_percept_circular(s, A_true, mu_true, sigma_true) for _ in range(n_mc)])
    expected_linear[idx] = np.mean(lin_samples)
    z = np.mean(np.exp(2j * np.pi * circ_samples))
    expected_circular[idx] = np.angle(z) / (2 * np.pi) % 1

bias_linear   = expected_linear - S
bias_circular = ((expected_circular - S) + 0.5) % 1 - 0.5   # circular difference

bias_theory             = analytical_bias(S_grid, A_true, mu_true, sigma_true)
expected_percept_theory = S_grid + bias_theory

print('done')

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

axes[0].plot(S_grid, S_grid, 'k--', lw=1, label='Identity')
axes[0].plot(S_grid, psi(S_grid, A_true, mu_true), 'gray', lw=2, ls=':', label='True ψ(s)')
axes[0].plot(S, expected_linear,   'bo', ms=6, label='MC linear mean')
axes[0].plot(S, expected_circular, 'gs', ms=6, label='MC circular mean')
axes[0].plot(S_grid, expected_percept_theory, 'r-', lw=2, label='Analytical (linear)')
axes[0].set_xlabel('Stimulus s'); axes[0].set_ylabel('E[ŝ | s]')
axes[0].set_title('Posterior mean: MC vs analytical')
axes[0].legend(fontsize=9); axes[0].grid(True, alpha=0.3)

axes[1].axhline(0, color='k', lw=0.8, ls='--')
axes[1].plot(S[1:], bias_linear[1:],   'bo', ms=6, label='MC linear bias')
axes[1].plot(S[1:], bias_circular[1:], 'gs', ms=6, label='MC circular bias')
axes[1].plot(S_grid, bias_theory, 'r-', lw=2, label='Analytical bias')
axes[1].set_xlabel('Stimulus s'); axes[1].set_ylabel('bias')
axes[1].set_title('Perceptual bias: MC vs analytical')
axes[1].legend(fontsize=9); axes[1].grid(True, alpha=0.3)

plt.tight_layout()
savefig('04_posterior_mean_and_bias.png')


# ── Part 3: Recover ψ via MLDS on Bayesian-observer data ──────────────────────

def simulate_trial_bayes(i, j, k, A, mu, sigma):
    """Each stimulus → linear posterior mean; compare interval magnitudes."""
    m1 = bayesian_percept_linear(S[i], A, mu, sigma)
    m2 = bayesian_percept_linear(S[j], A, mu, sigma)
    m3 = bayesian_percept_linear(S[k], A, mu, sigma)
    return int(abs(m1 - m2) > abs(m2 - m3))


data_bayes = []
for _ in range(n_repeats):
    for i, j, k in triplets:
        data_bayes.append([i, j, k, simulate_trial_bayes(i, j, k, A_true, mu_true, sigma_true)])
data_bayes = np.array(data_bayes)
print(f"{len(triplets)} triplets × {n_repeats} repeats = {len(data_bayes)} trials")

result_bayes  = minimize(neg_log_likelihood_mlds, np.linspace(0, 1, n_stim)[1:-1],
                         args=(data_bayes, sigma_true), method='L-BFGS-B')
psi_hat_bayes = reconstruct_psi(result_bayes.x)

plt.figure(figsize=(8, 6))
plt.plot(S_grid, rescale(psi(S_grid, A_true, mu_true)), 'gray', lw=2, ls=':', label='True ψ(s)')
plt.plot(S, psi_hat,       'g--s', ms=6, label='MLDS on MLDS data (Part 1)')
plt.plot(S, psi_hat_bayes, 'r--D', ms=6, label='MLDS on Bayesian-observer data (Part 3)')
plt.xlabel('Stimulus s')
plt.ylabel('Recovered ψ (normalized)')
plt.title('ψ recovery: MLDS model vs Bayesian observer')
plt.legend()
plt.grid(True, alpha=0.3)
savefig('05_psi_recovery_comparison.png')


# ── Part 4: Parameter recovery ─────────────────────────────────────────────────
#
# Recover (A, μ, σ) from a larger dataset using the analytical Bayesian observer model.
# A separate dataset with more repeats is generated here so Parts 1–3 stay fast.
#
# Percepts are approximately m_s ~ N(ψ̄_s, σ²/ψ'(s)²), independent across stimuli,
# giving a tractable triplet likelihood:
#
#   P(choice=1 | i,j,k) = Φ( (2ψ̄_j − ψ̄_i − ψ̄_k)
#                              / (σ · √(4/ψ'_j² + 1/ψ'_i² + 1/ψ'_k²)) )
#
# where ψ̄_s = s + bias(s; A, μ, σ)  and  ψ'_s = dpsi(s; A, μ).

n_repeats_recovery = 50
data_recovery = []
for _ in range(n_repeats_recovery):
    for i, j, k in triplets:
        data_recovery.append([i, j, k, simulate_trial_bayes(i, j, k, A_true, mu_true, sigma_true)])
data_recovery = np.array(data_recovery)
print(f"\nRecovery dataset: {len(triplets)} triplets × {n_repeats_recovery} repeats = {len(data_recovery)} trials")


ii_rec      = data_recovery[:, 0].astype(int)
jj_rec      = data_recovery[:, 1].astype(int)
kk_rec      = data_recovery[:, 2].astype(int)
choices_rec = data_recovery[:, 3]


def _nll_core(A, mu, sigma, ii, jj, kk, choices):
    psi_bar = expected_percept(S, A, mu, sigma)
    var_i   = percept_var(S[ii], A, mu, sigma)
    var_j   = percept_var(S[jj], A, mu, sigma)
    var_k   = percept_var(S[kk], A, mu, sigma)
    num   = 2 * psi_bar[jj] - psi_bar[ii] - psi_bar[kk]
    denom = np.sqrt(4 * var_j + var_i + var_k)
    p     = np.clip(norm.cdf(num / denom), 1e-10, 1 - 1e-10)
    return -(choices * np.log(p) + (1 - choices) * np.log(1 - p)).sum()


def nll_params(params, data):
    """Joint fit: (A, μ, σ) free."""
    A, mu, sigma = params
    return _nll_core(A, mu, sigma, ii_rec, jj_rec, kk_rec, choices_rec)


def nll_params_fixed_sigma(params, data):
    """Diagnostic fit: σ fixed at true value, only (A, μ) free."""
    A, mu = params
    return _nll_core(A, mu, sigma_true, ii_rec, jj_rec, kk_rec, choices_rec)


# Joint fit with multiple random restarts to avoid local minima
np.random.seed(0)
n_restarts = 10
bounds_joint = [(0.01, 0.99), (-0.5, 0.5), (1e-4, 0.2)]
best_nll, best_result = np.inf, None

for _ in range(n_restarts):
    x0 = [np.random.uniform(0.05, 0.8),
           np.random.uniform(-0.4, 0.4),
           np.random.uniform(0.005, 0.15)]
    res = minimize(nll_params, x0, args=(data_recovery,), method='L-BFGS-B',
                   bounds=bounds_joint, options={'ftol': 1e-12, 'gtol': 1e-8})
    if res.fun < best_nll:
        best_nll, best_result = res.fun, res

A_hat, mu_hat, sigma_hat = best_result.x
print(f"\n── Joint fit (A, μ, σ) — best of {n_restarts} restarts ──")
print(f"True:      A={A_true:.4f},  μ={mu_true:.4f},  σ={sigma_true:.4f}")
print(f"Recovered: A={A_hat:.4f},  μ={mu_hat:.4f},  σ={sigma_hat:.4f}")

# Diagnostic: fix σ and recover only (A, μ)
bounds_fixed = [(0.01, 0.99), (-0.5, 0.5)]
best_nll_f, best_result_f = np.inf, None

for _ in range(n_restarts):
    x0 = [np.random.uniform(0.05, 0.8), np.random.uniform(-0.4, 0.4)]
    res = minimize(nll_params_fixed_sigma, x0, args=(data_recovery,), method='L-BFGS-B',
                   bounds=bounds_fixed, options={'ftol': 1e-12, 'gtol': 1e-8})
    if res.fun < best_nll_f:
        best_nll_f, best_result_f = res.fun, res

A_hat_f, mu_hat_f = best_result_f.x
print(f"\n── Fixed-σ fit (A, μ only) — best of {n_restarts} restarts ──")
print(f"True:      A={A_true:.4f},  μ={mu_true:.4f}")
print(f"Recovered: A={A_hat_f:.4f},  μ={mu_hat_f:.4f}")

# ── Plots ──────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(2, 3, figsize=(15, 8))

for row, (A_h, mu_h, sig_h, label) in enumerate([
    (A_hat,   mu_hat,   sigma_hat,  f'Joint fit  (σ free → {sigma_hat:.4f})'),
    (A_hat_f, mu_hat_f, sigma_true, f'Fixed-σ fit (σ = {sigma_true})'),
]):
    axes[row, 0].plot(S_grid, rescale(psi(S_grid, A_true, mu_true)), 'k-',  lw=2, label='True')
    axes[row, 0].plot(S_grid, rescale(psi(S_grid, A_h,    mu_h)),    'r--', lw=2, label='Recovered')
    axes[row, 0].set_xlabel('s'); axes[row, 0].set_ylabel('ψ(s)')
    axes[row, 0].set_title(f'ψ — {label}')
    axes[row, 0].legend(fontsize=8); axes[row, 0].grid(True, alpha=0.3)

    axes[row, 1].plot(S_grid, prior_fn(S_grid, A_true, mu_true), 'k-',  lw=2, label='True')
    axes[row, 1].plot(S_grid, prior_fn(S_grid, A_h,    mu_h),    'r--', lw=2, label='Recovered')
    axes[row, 1].set_xlabel('s'); axes[row, 1].set_ylabel("p(s) = ψ'(s)")
    axes[row, 1].set_title("Prior = ψ'")
    axes[row, 1].legend(fontsize=8); axes[row, 1].grid(True, alpha=0.3)

    axes[row, 2].plot(S_grid, analytical_bias(S_grid, A_true, mu_true, sigma_true), 'k-',  lw=2, label='True')
    axes[row, 2].plot(S_grid, analytical_bias(S_grid, A_h,    mu_h,    sig_h),      'r--', lw=2, label='Recovered')
    axes[row, 2].axhline(0, color='gray', lw=0.8, ls='--')
    axes[row, 2].set_xlabel('s'); axes[row, 2].set_ylabel('bias')
    axes[row, 2].set_title('Perceptual bias')
    axes[row, 2].legend(fontsize=8); axes[row, 2].grid(True, alpha=0.3)

plt.tight_layout()
savefig('06_parameter_recovery.png')
