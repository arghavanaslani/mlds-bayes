"""
Bayesian Perceptual Model — Object-Oriented Architecture
=========================================================

Implements a static Bayesian observer model for psychophysical experiments.

Mathematical framework
----------------------
Encoding:  M = ψ(S) + σ·N(0, 1)
Posterior: p(s | r) ∝ N(r; ψ(s), σ) · p(s)
Decoding:  ŝ = E[s | r]  (posterior mean, L2 loss)

Wei & Stocker (2017) law: bias(s) ≈ d/ds[threshold(s)²]

Hahn & Wei (2024) decomposition (generalises W&S):
  bias(s) = (1/J(s))·d/ds[log p(s)]  +  (p+2)/4 · d/ds[1/J(s)]
            ───── attraction ─────       ───── repulsion ─────
  where J(s) = (ψ'(s))²/σ² and p is the loss exponent (p=2 → L2/mean).

Pipeline
--------
  Part 1 — MLDS triplet task → recover ψ̂ from simulated choices
  Part 2 — Bayesian inference → perceptual bias curve
  Part 3 — Verify Wei & Stocker law
  Part 4 — MLDS applied to a full Bayesian observer
  Part 5 — Hahn & Wei (2024) bias decomposition into attraction + repulsion

References
----------
  Maloney & Yang (2003)   — Maximum Likelihood Difference Scaling (MLDS)
  Wei & Stocker (2017)    — Lawful relation between bias and discriminability
  Hahn & Wei (2024)       — Unifying theory of perceptual biases
"""

from __future__ import annotations

import numpy as np
from abc import ABC, abstractmethod
from scipy.stats import norm, beta as beta_dist
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from typing import Callable, Optional


# =============================================================================
# Class 1: StimulusSpace
# =============================================================================

class StimulusSpace:
    """
    Represents the physical stimulus domain as a discretized 1D grid.

    The stimulus space defines the range of physical values (e.g. luminance,
    orientation, contrast) that can be presented to an observer. This class
    manages two grids: a coarse grid used for experimental trials, and a fine
    grid used for numerical integration in Bayesian inference.

    Parameters
    ----------
    s_min : float
        Minimum stimulus value.
    s_max : float
        Maximum stimulus value.
    n_stim : int
        Number of evenly spaced stimulus levels in the coarse grid.
    """

    def __init__(self, s_min: float, s_max: float, n_stim: int) -> None:
        self.s_min = s_min
        self.s_max = s_max
        self.n_stim = n_stim
        self._grid = np.linspace(s_min, s_max, n_stim)

    def get_grid(self) -> np.ndarray:
        """Return the coarse stimulus grid (n_stim equally spaced points)."""
        return self._grid.copy()

    def get_fine_grid(self, n: int = 200) -> np.ndarray:
        """
        Return a fine stimulus grid for smooth integration and plotting.

        Parameters
        ----------
        n : int
            Number of grid points (default: 200).
        """
        return np.linspace(self.s_min, self.s_max, n)

    def __repr__(self) -> str:
        return (f"StimulusSpace(s_min={self.s_min}, s_max={self.s_max}, "
                f"n_stim={self.n_stim})")


# =============================================================================
# Class 2: EncodingModel
# =============================================================================

class EncodingModel:
    """
    Sensory encoding function ψ with additive Gaussian noise.

    The encoding model transforms a physical stimulus s into an internal
    neural representation:

        M = ψ(s) + σ·N(0, 1)

    ψ captures the nonlinear mapping performed by the sensory system. Under
    optimal coding, ψ' is proportional to √p(s), so stimuli that occur more
    frequently are represented with finer resolution.

    Fisher information J(s) = (ψ'(s))² / σ² quantifies local discriminability.
    The discrimination threshold Δ(s) = 1/√J(s) is the smallest detectable
    stimulus difference near s.

    Parameters
    ----------
    psi_fn : Callable
        The encoding function ψ: R → R. Any callable is accepted.
    sigma : float
        Standard deviation of additive Gaussian noise in the encoded space.
    stimulus_space : StimulusSpace
        The stimulus domain this model operates on.
    """

    def __init__(self, psi_fn: Callable, sigma: float,
                 stimulus_space: StimulusSpace) -> None:
        self.psi_fn = psi_fn
        self.sigma = sigma
        self.stimulus_space = stimulus_space

    def encode(self, s: float | np.ndarray,
               rng: Optional[np.random.Generator] = None) -> float | np.ndarray:
        """
        Encode stimulus s into a noisy internal representation M = ψ(s) + N(0,σ).

        Parameters
        ----------
        s : float or array
            Stimulus value(s).
        rng : np.random.Generator, optional
            RNG for reproducibility. Uses np.random if None.
        """
        noise = (rng.normal(0, self.sigma, size=np.shape(s))
                 if rng is not None
                 else np.random.normal(0, self.sigma, size=np.shape(s)))
        return self.psi_fn(s) + noise

    def psi(self, s: float | np.ndarray) -> float | np.ndarray:
        """
        Return the noiseless encoding ψ(s).

        Parameters
        ----------
        s : float or array
        """
        return self.psi_fn(s)

    def fisher_information(self, s: float | np.ndarray,
                           ds: float = 1e-5) -> float | np.ndarray:
        """
        Compute Fisher information J(s) = (ψ'(s))² / σ².

        ψ'(s) is estimated via central finite differences.

        High J means the stimulus can be discriminated finely at s (steep ψ).
        Low J means coarse discrimination (flat ψ).

        Parameters
        ----------
        s : float or array
        ds : float
            Step size for numerical differentiation.
        """
        dpsi = (self.psi_fn(s + ds) - self.psi_fn(s - ds)) / (2.0 * ds)
        return dpsi ** 2 / self.sigma ** 2

    def discrimination_threshold(self, s: float | np.ndarray,
                                 ds: float = 1e-5) -> float | np.ndarray:
        """
        Compute the discrimination threshold Δ(s) = 1 / √J(s).

        The threshold is the smallest detectable stimulus change near s.
        Regions where ψ is steep (high J) have fine discrimination (low Δ).

        Parameters
        ----------
        s : float or array
        ds : float
        """
        return 1.0 / np.sqrt(self.fisher_information(s, ds=ds))

    def __repr__(self) -> str:
        name = getattr(self.psi_fn, '__name__', repr(self.psi_fn))
        return f"EncodingModel(psi={name}, sigma={self.sigma})"


# =============================================================================
# Class 3: Prior
# =============================================================================

class Prior:
    """
    Observer's prior beliefs over the stimulus distribution.

    In Bayesian perception, the prior p(s) encodes environmental statistics —
    which stimulus values are expected to occur. Under optimal coding,
    p(s) ∝ ψ'(s), meaning ψ allocates representational resources in proportion
    to stimulus frequency.

    A non-uniform prior causes systematic perceptual bias: the posterior mean
    is pulled toward high-prior regions. This is the mechanism behind the
    Wei & Stocker (2017) law.

    Parameters
    ----------
    prior_fn : Callable
        Non-negative callable representing the unnormalized prior density.
    stimulus_space : StimulusSpace
    """

    def __init__(self, prior_fn: Callable,
                 stimulus_space: StimulusSpace) -> None:
        self.prior_fn = prior_fn
        self.stimulus_space = stimulus_space

    def pdf(self, s: float | np.ndarray) -> float | np.ndarray:
        """
        Evaluate the prior density p(s).

        Parameters
        ----------
        s : float or array
        """
        return self.prior_fn(s)

    def plot(self, n: int = 200, ax=None, **kwargs) -> None:
        """
        Plot the prior density over the stimulus range.

        Parameters
        ----------
        n : int
            Number of points for the plot.
        ax : matplotlib Axes, optional
        """
        s_fine = self.stimulus_space.get_fine_grid(n)
        if ax is None:
            _, ax = plt.subplots()
        ax.plot(s_fine, self.pdf(s_fine), **kwargs)
        ax.set_xlabel("Stimulus s")
        ax.set_ylabel("Prior p(s)")
        ax.set_title("Prior distribution")

    @classmethod
    def uniform(cls, stimulus_space: StimulusSpace) -> "Prior":
        """
        Flat (uniform) prior: p(s) = 1.

        Under optimal coding, a uniform prior implies ψ should be linear.
        No stimulus value is preferred a priori.

        Parameters
        ----------
        stimulus_space : StimulusSpace
        """
        def prior_fn(s):
            return np.ones_like(np.asarray(s, dtype=float))
        prior_fn.__name__ = "uniform"
        return cls(prior_fn, stimulus_space)

    @classmethod
    def beta_mixture(cls, stimulus_space: StimulusSpace,
                     a: float, b: float,
                     weight: float = 0.5) -> "Prior":
        """
        Beta-mixture prior: p(s) = weight · (1 + Beta(s; a, b)).

        Models an environment with a unimodal preference (the Beta mode),
        blended with a uniform background. The peak is at (a-1)/(a+b-2)
        for a, b > 1.

        Parameters
        ----------
        stimulus_space : StimulusSpace
        a, b : float
            Beta distribution shape parameters.
        weight : float
            Mixing weight. 0 → flat prior; larger → stronger preference.
        """
        def prior_fn(s):
            return weight * (1.0 + beta_dist.pdf(np.asarray(s, dtype=float), a, b))
        prior_fn.__name__ = f"beta_mixture(a={a},b={b},w={weight})"
        return cls(prior_fn, stimulus_space)

    def __repr__(self) -> str:
        name = getattr(self.prior_fn, '__name__', repr(self.prior_fn))
        return f"Prior(fn={name})"


# =============================================================================
# Class 4: BayesianObserver
# =============================================================================

class BayesianObserver:
    """
    Bayesian observer combining sensory encoding and a prior for perception.

    Generative model:
        M = ψ(s) + σ·N(0, 1)          [encoding]
        p(s | r) ∝ N(r; ψ(s), σ)·p(s) [Bayesian inversion]
        ŝ = E[s | r]                    [decoding via posterior mean]

    Perceptual bias arises when the prior is non-uniform: the posterior mean
    is shifted toward regions of high prior density. The Wei & Stocker (2017)
    law states that this bias curve equals the derivative of the squared
    discrimination threshold: bias(s) ≈ d/ds[threshold(s)²].

    Parameters
    ----------
    encoding_model : EncodingModel
    prior : Prior
    n_integration : int
        Number of grid points for numerical integration (default: 400).
    rng : np.random.Generator, optional
    """

    def __init__(self, encoding_model: EncodingModel, prior: Prior,
                 n_integration: int = 400,
                 rng: Optional[np.random.Generator] = None) -> None:
        self.encoding = encoding_model
        self.prior = prior
        self.n_integration = n_integration
        self.rng = rng if rng is not None else np.random.default_rng()
        # Pre-compute the integration grid (shared across all calls)
        self._s_grid = encoding_model.stimulus_space.get_fine_grid(n_integration)

    def posterior(self, r: float) -> np.ndarray:
        """
        Compute the normalized posterior p(s | r) on the internal fine grid.

        p(s | r) ∝ N(r; ψ(s), σ) · p(s)

        Parameters
        ----------
        r : float
            Noisy internal measurement from the encoding model.

        Returns
        -------
        post : np.ndarray, shape (n_integration,)
            Normalized posterior evaluated at self._s_grid.
        """
        likelihood = norm.pdf(r, self.encoding.psi(self._s_grid),
                              self.encoding.sigma)
        prior_vals = self.prior.pdf(self._s_grid)
        post = likelihood * prior_vals
        Z = np.trapz(post, self._s_grid)
        return post / (Z + 1e-300)

    def estimate(self, r: float, loss: str = 'L2') -> float:
        """
        Decode a measurement r into a stimulus estimate via the posterior.

        Parameters
        ----------
        r : float
            Noisy measurement from the encoding step.
        loss : {'L2', 'L1', 'L0'}
            Decoding loss:
            - 'L2' : posterior mean  (minimises MSE)
            - 'L1' : posterior median (minimises MAE)
            - 'L0' : MAP estimate    (maximises posterior)
        """
        post = self.posterior(r)
        if loss == 'L2':
            return float(np.trapz(self._s_grid * post, self._s_grid))
        elif loss == 'L1':
            ds = self._s_grid[1] - self._s_grid[0]
            cdf = np.cumsum(post) * ds
            idx = np.searchsorted(cdf, 0.5)
            return float(self._s_grid[min(idx, len(self._s_grid) - 1)])
        elif loss == 'L0':
            return float(self._s_grid[np.argmax(post)])
        else:
            raise ValueError(f"Unknown loss '{loss}'. Use 'L2', 'L1', or 'L0'.")

    def simulate_estimate(self, s_true: float, loss: str = 'L2') -> float:
        """
        Full encode → decode pipeline for a single simulated trial.

        1. Encode:  r = ψ(s_true) + N(0, σ)
        2. Decode:  ŝ = estimate(r)

        Parameters
        ----------
        s_true : float
            The true presented stimulus.
        loss : str
        """
        r = float(self.encoding.psi(s_true)) + float(
            self.rng.normal(0, self.encoding.sigma))
        return self.estimate(r, loss=loss)

    def _collect_estimates(self, stimuli: np.ndarray,
                           n_rep: int, loss: str) -> tuple[np.ndarray, np.ndarray]:
        """
        Collect bias and threshold in a single pass over all stimuli.

        Returns
        -------
        bias : np.ndarray
        threshold : np.ndarray
        """
        bias = np.empty(len(stimuli))
        threshold = np.empty(len(stimuli))
        for i, s in enumerate(stimuli):
            estimates = np.array([self.simulate_estimate(s, loss=loss)
                                  for _ in range(n_rep)])
            bias[i] = float(np.mean(estimates)) - s
            threshold[i] = float(np.std(estimates))
        return bias, threshold

    def compute_bias(self, stimuli: np.ndarray, n_rep: int = 500,
                     loss: str = 'L2') -> np.ndarray:
        """
        Compute the perceptual bias curve: bias(s) = E[ŝ | s] − s.

        Positive bias → overestimation; negative → underestimation.
        The sign and shape are predicted by the prior via the Wei & Stocker law.

        Parameters
        ----------
        stimuli : np.ndarray
        n_rep : int
            Monte Carlo repetitions per stimulus.
        loss : str
        """
        bias, _ = self._collect_estimates(stimuli, n_rep, loss)
        return bias

    def compute_threshold(self, stimuli: np.ndarray, n_rep: int = 500,
                          loss: str = 'L2') -> np.ndarray:
        """
        Compute the discrimination threshold: threshold(s) = std(ŝ | s).

        This is a behavioral measure of precision. Under Fisher information
        theory, threshold ≈ 1/√J(s) in the asymptotic regime.

        Parameters
        ----------
        stimuli : np.ndarray
        n_rep : int
        loss : str
        """
        _, threshold = self._collect_estimates(stimuli, n_rep, loss)
        return threshold

    def check_wei_stocker_law(self, stimuli: np.ndarray,
                               n_rep: int = 500,
                               loss: str = 'L2') -> tuple[np.ndarray, np.ndarray]:
        """
        Verify the Wei & Stocker (2017) law:  bias(s) ≈ d/ds[threshold(s)²].

        This is a model-free test: both bias and threshold are measured
        independently from behavior, and the law predicts they are related
        through the derivative of the squared threshold.

        The law is a direct consequence of Bayesian inference under the
        Gaussian noise model. It holds regardless of the specific form of ψ
        or the prior, making it a strong and general prediction.

        Parameters
        ----------
        stimuli : np.ndarray
        n_rep : int
        loss : str

        Returns
        -------
        bias : np.ndarray
            Measured bias at each stimulus.
        predicted_bias : np.ndarray
            Prediction from d(threshold²)/dθ.
        """
        bias, threshold = self._collect_estimates(stimuli, n_rep, loss)
        predicted_bias = np.gradient(threshold ** 2, stimuli)
        return bias, predicted_bias

    # -----------------------------------------------------------------
    # Hahn & Wei (2024) analytical bias decomposition
    # -----------------------------------------------------------------

    def decompose_bias(self, stimuli: np.ndarray, p: float = 2.0,
                       ds: float = 1e-5) -> dict:
        """
        Additive decomposition of perceptual bias (Hahn & Wei 2024, Eq. 2–3).

        The total bias of the Bayesian optimal estimate under an L^p loss
        decomposes into two independent terms:

          bias(s) ≈  attraction(s)  +  repulsion(s)

        Prior attraction — pulls the estimate toward high-prior regions:
          attraction(s) = (1/J(s)) · d/ds[log p_prior(s)]

        Likelihood repulsion — pushes the estimate away from regions of
        high encoding precision (steep ψ → narrow likelihood → asymmetry):
          repulsion(s) = coeff(p) · d/ds[1/J(s)]
        where coeff(p) = (p+2)/4 for p ≥ 1 and coeff = 1/4 for MAP (p → 0).

        Key insight: attraction is independent of the loss function, while
        repulsion scales with the exponent p. This explains why experiments
        can observe either attractive or repulsive biases depending on which
        component dominates.

        Parameters
        ----------
        stimuli : np.ndarray
            Stimulus values at which to evaluate the decomposition. Should
            be a fine, evenly-spaced grid for accurate gradient computation.
        p : float
            Loss function exponent. p=2 → posterior mean (L2, standard);
            p=1 → posterior median (L1); p→0 → MAP.
        ds : float
            Step size for numerical differentiation of ψ.

        Returns
        -------
        dict with keys:
            'stimuli'         : np.ndarray — the input grid
            'attraction'      : np.ndarray — prior attraction component
            'repulsion'       : np.ndarray — likelihood repulsion component
            'total_predicted' : np.ndarray — attraction + repulsion
            'J'               : np.ndarray — Fisher information J(s)
            'resources'       : np.ndarray — √J(s) (encoding resource allocation)
            'p'               : float      — the loss exponent used

        References
        ----------
        Hahn & Wei (2024), Eqs. 2–3 and the P/P ratio rule (Eq. 4).
        """
        J = self.encoding.fisher_information(stimuli, ds=ds)
        prior_vals = self.prior.pdf(stimuli)

        # ── Prior attraction: (1/J) · d/ds[log p(s)] ────────────────────
        log_prior = np.log(np.maximum(prior_vals, 1e-300))
        d_log_prior = np.gradient(log_prior, stimuli)
        attraction = (1.0 / J) * d_log_prior

        # ── Likelihood repulsion: coeff · d/ds[1/J(s)] ──────────────────
        inv_J = 1.0 / J
        d_inv_J = np.gradient(inv_J, stimuli)
        # MAP (p→0) has a special coefficient (Eq. 3 vs Eq. 2)
        coeff = 0.25 if p < 0.5 else (p + 2.0) / 4.0
        repulsion = coeff * d_inv_J

        total = attraction + repulsion

        return {
            'stimuli':         stimuli,
            'attraction':      attraction,
            'repulsion':       repulsion,
            'total_predicted': total,
            'J':               J,
            'resources':       np.sqrt(J),
            'p':               p,
        }

    def pp_ratio(self, stimuli: np.ndarray, p: float = 2.0,
                 ds: float = 1e-5) -> np.ndarray:
        """
        P/P ratio rule for predicting the sign of the total bias
        (Hahn & Wei 2024, Eq. 4).

        Defines a ratio function:
          Q(s) = d/ds[ p_prior(s) / J(s)^((p+2)/4) ]

        When Q(s) > 0, the total bias is predicted to be positive
        (attractive toward higher-prior regions dominate).
        When Q(s) < 0, the total bias is predicted to be negative
        (repulsive effects dominate).

        The denominator captures the encoding precision raised to a
        loss-dependent power. The rule formalizes the intuition that
        whichever of prior density or encoding precision varies more
        rapidly determines the direction of the bias.

        Parameters
        ----------
        stimuli : np.ndarray
        p : float
            Loss function exponent.
        ds : float

        Returns
        -------
        Q : np.ndarray
            The derivative of the P/P ratio; its sign predicts bias direction.
        """
        J = self.encoding.fisher_information(stimuli, ds=ds)
        prior_vals = self.prior.pdf(stimuli)
        power = (p + 2.0) / 4.0
        ratio = prior_vals / np.power(J, power)
        return np.gradient(ratio, stimuli)

    def __repr__(self) -> str:
        return f"BayesianObserver(encoding={self.encoding}, prior={self.prior})"


# =============================================================================
# Class 5: ExperimentTask (abstract base) + MLDSTask + TwoAFCTask
# =============================================================================

class ExperimentTask(ABC):
    """
    Abstract base class for psychophysical experiment tasks.

    A task encapsulates: how trials are generated, how the observer responds,
    and how data are analysed to recover a perceptual property (e.g., ψ̂).

    Parameters
    ----------
    observer : BayesianObserver
    stimulus_space : StimulusSpace
    n_trials : int
        Default number of trials when run() is called.
    rng : np.random.Generator, optional
    """

    def __init__(self, observer: BayesianObserver,
                 stimulus_space: StimulusSpace,
                 n_trials: int = 300,
                 rng: Optional[np.random.Generator] = None) -> None:
        self.observer = observer
        self.stimulus_space = stimulus_space
        self.n_trials = n_trials
        self.rng = rng if rng is not None else np.random.default_rng()

    @abstractmethod
    def generate_trial(self) -> tuple:
        """Generate a single trial (stimulus indices or values)."""

    @abstractmethod
    def simulate_trial(self, trial: tuple) -> int:
        """Simulate the observer's response to a trial. Returns an int (0 or 1)."""

    def run(self, n_trials: Optional[int] = None) -> np.ndarray:
        """
        Run the full experiment and return collected data.

        Parameters
        ----------
        n_trials : int, optional
            Overrides self.n_trials if provided.

        Returns
        -------
        data : np.ndarray, shape (n_trials, trial_width + 1)
            Each row is (*trial, response).
        """
        n = n_trials if n_trials is not None else self.n_trials
        data = []
        for _ in range(n):
            trial = self.generate_trial()
            response = self.simulate_trial(trial)
            data.append((*trial, response))
        return np.array(data, dtype=float)

    @abstractmethod
    def fit(self, data: np.ndarray) -> np.ndarray:
        """Fit model to data and return estimated ψ̂ or threshold curve."""


class MLDSTask(ExperimentTask):
    """
    Maximum Likelihood Difference Scaling (MLDS) — triplet task.

    On each trial the observer sees three stimuli s_i < s_j < s_k and judges
    which adjacent pair appears perceptually more different:
      choice = 1  if  |ψ(j)−ψ(i)| > |ψ(k)−ψ(j)|
      choice = 0  otherwise

    Decision variable (Maloney & Yang 2003, eq. 2):
        D(i,j,k) = (ψ(j)−ψ(i)) − (ψ(k)−ψ(j)) + ε,  ε ~ N(0, σ)

    MLE recovers ψ̂(s₂), …, ψ̂(s_{N-1}) with boundary conditions
    ψ̂(s₁)=0, ψ̂(s_N)=1.

    The `use_bayesian_observer` flag switches between two simulation modes:
      - False (Part 1): raw ψ values + decision noise; recovers ψ directly.
      - True  (Part 4): full encode-decode pipeline; the MLDS then captures
        the observer's effective perceptual scale including Bayesian bias.

    Parameters
    ----------
    observer : BayesianObserver
    stimulus_space : StimulusSpace
    n_trials : int
    sigma_decision : float
        Decision noise σ used both in simulation (False mode) and MLE fitting.
    use_bayesian_observer : bool
        If True, simulate percepts via the full Bayesian encode-decode pipeline.
    rng : np.random.Generator, optional
    """

    def __init__(self, observer: BayesianObserver,
                 stimulus_space: StimulusSpace,
                 n_trials: int = 300,
                 sigma_decision: float = 0.1,
                 use_bayesian_observer: bool = False,
                 rng: Optional[np.random.Generator] = None) -> None:
        super().__init__(observer, stimulus_space, n_trials, rng)
        self.sigma_decision = sigma_decision
        self.use_bayesian_observer = use_bayesian_observer
        self._indices = list(range(stimulus_space.n_stim))

    def generate_trial(self) -> tuple[int, int, int]:
        """
        Sample 3 distinct stimulus indices i < j < k uniformly at random.
        """
        return tuple(sorted(
            self.rng.choice(self._indices, size=3, replace=False).tolist()
        ))

    def simulate_trial(self, trial: tuple[int, int, int]) -> int:
        """
        Simulate observer response to triplet (i, j, k).

        Returns 1 if the first interval (i,j) seems more different, 0 otherwise.
        """
        i, j, k = trial
        S = self.stimulus_space.get_grid()

        if self.use_bayesian_observer:
            # Full Bayesian: each presentation triggers encode → decode
            si = self.observer.simulate_estimate(S[i])
            sj = self.observer.simulate_estimate(S[j])
            sk = self.observer.simulate_estimate(S[k])
            decision_variable = abs(si - sj) - abs(sj - sk)
        else:
            # Simple MLDS: ψ values + Gaussian decision noise
            psi = self.observer.encoding.psi(S)
            d1 = abs(psi[i] - psi[j])
            d2 = abs(psi[j] - psi[k])
            decision_variable = (d1 - d2
                                 + self.rng.normal(0, self.sigma_decision))

        return int(decision_variable > 0)

    def fit(self, data: np.ndarray, method: str = 'L-BFGS-B') -> np.ndarray:
        """
        Recover the perceptual scale ψ̂ by maximum likelihood.

        Free parameters: ψ(s₂), …, ψ(s_{N-1}), with ψ(s₁)=0, ψ(s_N)=1.

        Log-likelihood (Maloney & Yang 2003, eq. 6–7):
            ℓ = Σ_t [ R_t · log Φ(Δ_t/σ) + (1−R_t) · log(1 − Φ(Δ_t/σ)) ]
        where Δ_t = (ψ(j)−ψ(i)) − (ψ(k)−ψ(j)) and Φ is the normal CDF.

        Parameters
        ----------
        data : np.ndarray, shape (n_trials, 4)
            Columns: [i, j, k, choice].
        method : str
            Optimization method for scipy.optimize.minimize.

        Returns
        -------
        psi_hat : np.ndarray, shape (n_stim,)
        """
        n = self.stimulus_space.n_stim
        sigma = self.sigma_decision

        def reconstruct(params: np.ndarray) -> np.ndarray:
            psi = np.empty(n)
            psi[0] = 0.0
            psi[-1] = 1.0
            psi[1:-1] = params
            return psi

        def neg_log_likelihood(params: np.ndarray) -> float:
            psi = reconstruct(params)
            ll = 0.0
            for row in data:
                i, j, k, choice = int(row[0]), int(row[1]), int(row[2]), int(row[3])
                d1 = abs(psi[i] - psi[j])
                d2 = abs(psi[j] - psi[k])
                p = np.clip(norm.cdf((d1 - d2) / sigma), 1e-10, 1.0 - 1e-10)
                ll += np.log(p) if choice == 1 else np.log(1.0 - p)
            return -ll

        x0 = np.linspace(0.0, 1.0, n)[1:-1]
        result = minimize(neg_log_likelihood, x0, method=method)
        return reconstruct(result.x)


class TwoAFCTask(ExperimentTask):
    """
    Two-Alternative Forced Choice (2AFC) discrimination task.

    On each trial:
      - Interval 1: reference stimulus s_ref
      - Interval 2: comparison stimulus s_ref + Δ (Δ drawn from delta_values)

    The observer encodes both stimuli independently and picks the interval
    with the larger internal measurement. By varying Δ, we trace out a
    psychometric function and estimate the JND (just-noticeable difference,
    i.e. the threshold) at each reference.

    The psychometric function is:
        P(choose comparison | Δ) = Φ((Δ − μ) / threshold)

    where μ is the point of subjective equality (PSE) bias.

    Parameters
    ----------
    observer : BayesianObserver
    stimulus_space : StimulusSpace
    n_trials : int
    delta_values : np.ndarray, optional
        Array of signed Δ offsets to test. Default: 15 values in [−0.2, 0.2].
    rng : np.random.Generator, optional
    """

    def __init__(self, observer: BayesianObserver,
                 stimulus_space: StimulusSpace,
                 n_trials: int = 500,
                 delta_values: Optional[np.ndarray] = None,
                 rng: Optional[np.random.Generator] = None) -> None:
        super().__init__(observer, stimulus_space, n_trials, rng)
        if delta_values is None:
            self.delta_values = np.linspace(-0.2, 0.2, 15)
        else:
            self.delta_values = np.asarray(delta_values)
        self._ref_grid = stimulus_space.get_grid()

    def generate_trial(self) -> tuple[float, float]:
        """
        Sample a reference and comparison stimulus.

        Returns (s_ref, s_comp) where s_comp = s_ref + Δ, clipped to [s_min, s_max].
        """
        s_ref = float(self.rng.choice(self._ref_grid))
        delta = float(self.rng.choice(self.delta_values))
        s_comp = float(np.clip(s_ref + delta,
                               self.stimulus_space.s_min,
                               self.stimulus_space.s_max))
        return (s_ref, s_comp)

    def simulate_trial(self, trial: tuple[float, float]) -> int:
        """
        Simulate 2AFC response.

        The observer independently encodes both intervals and picks the one
        with the larger internal measurement.

        Returns 1 if comparison > reference, 0 otherwise.
        """
        s_ref, s_comp = trial
        sigma = self.observer.encoding.sigma
        r_ref = (float(self.observer.encoding.psi(s_ref))
                 + float(self.rng.normal(0, sigma)))
        r_comp = (float(self.observer.encoding.psi(s_comp))
                  + float(self.rng.normal(0, sigma)))
        return int(r_comp > r_ref)

    def fit(self, data: np.ndarray,
            ref_stimuli: Optional[np.ndarray] = None,
            atol: float = 1e-6) -> dict:
        """
        Fit a cumulative Gaussian psychometric function per reference stimulus.

        For each reference, collects all trials where s_ref matches, then fits:
            P(choose comparison) = Φ((Δ − μ) / threshold)
        by MLE to recover the threshold (σ of the psychometric function) and
        the PSE offset (μ).

        Parameters
        ----------
        data : np.ndarray, shape (n_trials, 3)
            Columns: [s_ref, s_comp, choice].
        ref_stimuli : np.ndarray, optional
            Reference values to fit at. Defaults to stimulus_space grid.
        atol : float
            Absolute tolerance for matching s_ref values in data.

        Returns
        -------
        dict with keys:
            'ref_stimuli' : np.ndarray
            'thresholds'  : np.ndarray  — JND at each reference (NaN if too few trials)
            'biases'      : np.ndarray  — PSE offset at each reference
        """
        if ref_stimuli is None:
            ref_stimuli = self.stimulus_space.get_grid()

        thresholds, biases = [], []
        for s_ref in ref_stimuli:
            mask = np.abs(data[:, 0] - s_ref) < atol
            sub = data[mask]
            if len(sub) < 5:
                thresholds.append(np.nan)
                biases.append(np.nan)
                continue

            deltas = sub[:, 1] - sub[:, 0]
            choices = sub[:, 2].astype(int)

            def neg_ll(params: np.ndarray) -> float:
                mu, sigma_p = params
                p = np.clip(norm.cdf((deltas - mu) / (abs(sigma_p) + 1e-8)),
                            1e-10, 1.0 - 1e-10)
                return -float(np.sum(choices * np.log(p)
                                     + (1 - choices) * np.log(1 - p)))

            res = minimize(neg_ll, [0.0, 0.05],
                           bounds=[(-0.5, 0.5), (1e-4, 0.5)],
                           method='L-BFGS-B')
            biases.append(res.x[0])
            thresholds.append(abs(res.x[1]))

        return {
            'ref_stimuli': np.array(ref_stimuli),
            'thresholds':  np.array(thresholds),
            'biases':      np.array(biases),
        }


# =============================================================================
# Factory functions
# =============================================================================

def make_psi(shape: str, stimulus_space: StimulusSpace,
             **kwargs) -> Callable:
    """
    Return a named ψ function (callable) normalized to [0, 1] on the stimulus range.

    Parameters
    ----------
    shape : {'linear', 'log', 'beta_cdf', 'power'}
        Name of the ψ function.
    stimulus_space : StimulusSpace
        Used to normalize output to [0, 1].
    **kwargs
        Shape-specific parameters:
        - 'log'     : offset (default 0.1)
        - 'beta_cdf': a (default 10), b (default 3), mixture (default True)
                      If mixture=True:  ψ(s) = 0.5·(s + BetaCDF(s; a, b))
                      If mixture=False: ψ(s) = BetaCDF(s; a, b)
        - 'power'   : gamma (default 0.5), ψ(s) = s^gamma

    Returns
    -------
    psi_fn : Callable
        A function s → ψ(s) returning values in [0, 1].
    """
    s_min = stimulus_space.s_min
    s_max = stimulus_space.s_max

    if shape == 'linear':
        def psi_fn(s):
            return (np.asarray(s, dtype=float) - s_min) / (s_max - s_min)
        psi_fn.__name__ = 'linear'

    elif shape == 'log':
        offset = kwargs.get('offset', 0.1)
        raw_min = float(np.log(s_min + offset))
        raw_max = float(np.log(s_max + offset))

        def psi_fn(s):
            raw = np.log(np.asarray(s, dtype=float) + offset)
            return (raw - raw_min) / (raw_max - raw_min)
        psi_fn.__name__ = f'log(offset={offset})'

    elif shape == 'beta_cdf':
        a = kwargs.get('a', 10)
        b = kwargs.get('b', 3)
        mixture = kwargs.get('mixture', True)
        # Precompute normalization constants from the fine grid
        s_norm = stimulus_space.get_fine_grid(500)
        v_norm = (0.5 * (s_norm + beta_dist.cdf(s_norm, a, b)) if mixture
                  else beta_dist.cdf(s_norm, a, b))
        v_min = float(v_norm.min())
        v_max = float(v_norm.max())

        def psi_fn(s):
            s = np.asarray(s, dtype=float)
            raw = (0.5 * (s + beta_dist.cdf(s, a, b)) if mixture
                   else beta_dist.cdf(s, a, b))
            return (raw - v_min) / (v_max - v_min + 1e-300)
        psi_fn.__name__ = f'beta_cdf(a={a},b={b},mixture={mixture})'

    elif shape == 'power':
        gamma = kwargs.get('gamma', 0.5)

        def psi_fn(s):
            s_unit = (np.asarray(s, dtype=float) - s_min) / (s_max - s_min)
            return s_unit ** gamma
        psi_fn.__name__ = f'power(gamma={gamma})'

    else:
        raise ValueError(
            f"Unknown shape '{shape}'. "
            f"Choose from: 'linear', 'log', 'beta_cdf', 'power'."
        )

    return psi_fn


def make_prior(shape: str, stimulus_space: StimulusSpace,
               **kwargs) -> Prior:
    """
    Return a named Prior object.

    Parameters
    ----------
    shape : {'uniform', 'beta', 'gaussian'}
    stimulus_space : StimulusSpace
    **kwargs
        - 'beta'     : a (default 2), b (default 2), weight (default 0.5)
        - 'gaussian' : mu (default 0.5), sigma (default 0.2)

    Returns
    -------
    prior : Prior
    """
    if shape == 'uniform':
        return Prior.uniform(stimulus_space)

    elif shape == 'beta':
        a = kwargs.get('a', 2)
        b = kwargs.get('b', 2)
        weight = kwargs.get('weight', 0.5)
        return Prior.beta_mixture(stimulus_space, a=a, b=b, weight=weight)

    elif shape == 'gaussian':
        mu = kwargs.get('mu', 0.5)
        sigma_p = kwargs.get('sigma', 0.2)

        def prior_fn(s):
            return np.exp(-0.5 * ((np.asarray(s, dtype=float) - mu) / sigma_p) ** 2)
        prior_fn.__name__ = f'gaussian(mu={mu},sigma={sigma_p})'
        return Prior(prior_fn, stimulus_space)

    else:
        raise ValueError(
            f"Unknown shape '{shape}'. "
            f"Choose from: 'uniform', 'beta', 'gaussian'."
        )


# =============================================================================
# Prior & posterior visualisation
# =============================================================================

def plot_prior_posterior(
        observer: BayesianObserver,
        example_stimuli: Optional[np.ndarray] = None,
        noiseless: bool = True,
        rng: Optional[np.random.Generator] = None,
) -> plt.Figure:
    """
    Two-panel figure showing the prior and a set of example posteriors.

    Left panel  — Prior p(s) (normalised to unit area for display).
    Right panel — Posterior p(s | r) for each example stimulus.
                  By default r = ψ(s) (noiseless) so the posterior is centred
                  cleanly on the true stimulus; set ``noiseless=False`` to draw
                  one noisy measurement per example instead.

    Vertical dashed lines mark the true stimulus positions so you can see
    how the prior pulls the posterior mean away from the physical value.

    Parameters
    ----------
    observer : BayesianObserver
        The observer whose encoding and prior are used.
    example_stimuli : np.ndarray, optional
        Stimulus values at which to evaluate the posterior.
        Defaults to 5 values spread across 10–90 % of the stimulus range.
    noiseless : bool
        If True, use r = ψ(s) exactly (clean illustration).
        If False, draw r ~ N(ψ(s), σ) for each example.
    rng : np.random.Generator, optional
        Used only when ``noiseless=False``.

    Returns
    -------
    fig : matplotlib Figure
    """
    space  = observer.encoding.stimulus_space
    s_fine = space.get_fine_grid(400)

    if example_stimuli is None:
        span = space.s_max - space.s_min
        example_stimuli = np.linspace(space.s_min + 0.1 * span,
                                      space.s_max - 0.1 * span, 5)

    fig, (ax_prior, ax_post) = plt.subplots(1, 2, figsize=(11, 4))
    fig.suptitle("Prior and Posterior Distributions", fontsize=12)

    # ── Left panel: prior ────────────────────────────────────────────────────
    prior_vals = observer.prior.pdf(s_fine)
    Z_prior    = np.trapz(prior_vals, s_fine)
    prior_norm = prior_vals / (Z_prior + 1e-300)

    ax_prior.plot(s_fine, prior_norm, color='steelblue', lw=2)
    ax_prior.fill_between(s_fine, prior_norm, alpha=0.25, color='steelblue')
    ax_prior.set_xlabel("Stimulus s")
    ax_prior.set_ylabel("Density")
    ax_prior.set_title("Prior  $p(s)$")
    ax_prior.set_xlim(space.s_min, space.s_max)
    ax_prior.set_ylim(bottom=0)

    # ── Right panel: posteriors ───────────────────────────────────────────────
    colors = plt.cm.plasma(np.linspace(0.15, 0.85, len(example_stimuli)))
    _rng   = rng if rng is not None else np.random.default_rng()

    for s_ex, col in zip(example_stimuli, colors):
        if noiseless:
            r = float(observer.encoding.psi(s_ex))
        else:
            r = (float(observer.encoding.psi(s_ex))
                 + float(_rng.normal(0, observer.encoding.sigma)))

        # Posterior evaluated on the observer's internal grid, then
        # interpolated onto the fine grid for smooth plotting.
        post        = observer.posterior(r)
        post_fine   = np.interp(s_fine, observer._s_grid, post)
        Z_post      = np.trapz(post_fine, s_fine)
        post_fine  /= (Z_post + 1e-300)

        ax_post.plot(s_fine, post_fine, color=col, lw=1.8,
                     label=f"$s={s_ex:.2f}$")
        # Dashed vertical line marks the true stimulus
        ax_post.axvline(s_ex, color=col, lw=0.8, ls='--', alpha=0.65)

    ax_post.set_xlabel("Stimulus s")
    ax_post.set_ylabel("Density")
    ax_post.set_title("Posterior  $p(s\\,|\\,r)$  for example stimuli")
    ax_post.set_xlim(space.s_min, space.s_max)
    ax_post.set_ylim(bottom=0)
    ax_post.legend(fontsize=8, loc='upper left')

    fig.tight_layout()
    return fig


# =============================================================================
# Hahn & Wei (2024) decomposition plot
# =============================================================================

def plot_hahn_decomposition(
        observer: BayesianObserver,
        stimuli: np.ndarray,
        simulated_bias: Optional[np.ndarray] = None,
        p_values: Optional[list] = None,
) -> plt.Figure:
    """
    Five-panel figure reproducing the style of Hahn & Wei (2024) Figs. 1d–i.

    Panels: Prior | Resources (√J) | Attraction | Repulsion | Total bias

    If ``simulated_bias`` is provided, the measured (Monte-Carlo) bias is
    overlaid on the "Total" panel for comparison with the analytical
    prediction. If ``p_values`` contains more than one exponent, both
    Repulsion and Total panels show one curve per exponent so you can see
    how only the repulsive component scales with the loss function.

    Parameters
    ----------
    observer : BayesianObserver
    stimuli : np.ndarray
        Fine, evenly-spaced grid (e.g. 50+ points).
    simulated_bias : np.ndarray, optional
        Measured bias from observer.compute_bias() for overlay.
    p_values : list of float, optional
        Loss exponents to compare. Default: [0, 2, 4, 8].

    Returns
    -------
    fig : matplotlib Figure
    """
    if p_values is None:
        p_values = [0, 2, 4, 8]

    space = observer.encoding.stimulus_space

    # Compute decomposition at each loss exponent p
    decomps = {p: observer.decompose_bias(stimuli, p=p) for p in p_values}
    # Use the first decomposition for prior / resources (same for all p)
    d0 = decomps[p_values[0]]

    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(p_values)))

    fig, axes = plt.subplots(1, 5, figsize=(22, 3.8))
    fig.suptitle("Hahn & Wei (2024) — Bias Decomposition", fontsize=13)

    # ── Panel 1: Prior ────────────────────────────────────────────────────
    ax = axes[0]
    prior_vals = observer.prior.pdf(stimuli)
    Z = np.trapz(prior_vals, stimuli)
    ax.plot(stimuli, prior_vals / (Z + 1e-300), color='steelblue', lw=2)
    ax.fill_between(stimuli, prior_vals / (Z + 1e-300),
                    alpha=0.2, color='steelblue')
    ax.set_xlabel("Stimulus $s$")
    ax.set_ylabel("Density")
    ax.set_title("Prior $p(s)$")
    ax.set_xlim(space.s_min, space.s_max)
    ax.set_ylim(bottom=0)

    # ── Panel 2: Resources √J ────────────────────────────────────────────
    ax = axes[1]
    ax.plot(stimuli, d0['resources'], color='darkorange', lw=2)
    ax.fill_between(stimuli, d0['resources'], alpha=0.2, color='darkorange')
    ax.set_xlabel("Stimulus $s$")
    ax.set_ylabel("$\\sqrt{J(s)}$")
    ax.set_title("Encoding resources $\\sqrt{J}$")
    ax.set_xlim(space.s_min, space.s_max)
    ax.set_ylim(bottom=0)

    # ── Panel 3: Attraction (independent of p) ────────────────────────────
    ax = axes[2]
    ax.plot(stimuli, d0['attraction'], color='green', lw=2)
    ax.axhline(0, color='k', lw=0.6, ls='--')
    ax.set_xlabel("Stimulus $s$")
    ax.set_ylabel("Bias")
    ax.set_title("Prior attraction")
    ax.set_xlim(space.s_min, space.s_max)

    # ── Panel 4: Repulsion (depends on p) ─────────────────────────────────
    ax = axes[3]
    for p, col in zip(p_values, colors):
        d = decomps[p]
        label = f"$p={p}$" if p > 0 else "MAP ($p\\to 0$)"
        ax.plot(stimuli, d['repulsion'], color=col, lw=1.8, label=label)
    ax.axhline(0, color='k', lw=0.6, ls='--')
    ax.set_xlabel("Stimulus $s$")
    ax.set_ylabel("Bias")
    ax.set_title("Likelihood repulsion")
    ax.legend(fontsize=7, loc='best')
    ax.set_xlim(space.s_min, space.s_max)

    # ── Panel 5: Total (attraction + repulsion) ───────────────────────────
    ax = axes[4]
    for p, col in zip(p_values, colors):
        d = decomps[p]
        label = f"$p={p}$" if p > 0 else "MAP ($p\\to 0$)"
        ax.plot(stimuli, d['total_predicted'], color=col, lw=1.8, label=label)
    if simulated_bias is not None:
        ax.plot(stimuli, simulated_bias, 'ko', ms=4, alpha=0.6,
                label="Simulated ($p=2$)")
    ax.axhline(0, color='k', lw=0.6, ls='--')
    ax.set_xlabel("Stimulus $s$")
    ax.set_ylabel("Bias")
    ax.set_title("Total bias = attr. + rep.")
    ax.legend(fontsize=7, loc='best')
    ax.set_xlim(space.s_min, space.s_max)

    fig.tight_layout()
    return fig


def plot_pp_ratio(
        observer: BayesianObserver,
        stimuli: np.ndarray,
        simulated_bias: Optional[np.ndarray] = None,
        p: float = 2.0,
) -> plt.Figure:
    """
    Two-panel figure testing the P/P ratio rule (Hahn & Wei 2024, Eq. 4).

    Left panel  — the P/P ratio Q(s) and its sign.
    Right panel — simulated bias overlaid with sign-predicted direction.

    The rule predicts that wherever Q(s) > 0 the bias is positive
    (attraction wins), and wherever Q(s) < 0 the bias is negative
    (repulsion wins).

    Parameters
    ----------
    observer : BayesianObserver
    stimuli : np.ndarray
    simulated_bias : np.ndarray, optional
    p : float
        Loss exponent.

    Returns
    -------
    fig : matplotlib Figure
    """
    Q = observer.pp_ratio(stimuli, p=p)
    decomp = observer.decompose_bias(stimuli, p=p)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    fig.suptitle(f"P/P Ratio Rule (Hahn & Wei 2024)  —  $p={p}$", fontsize=12)

    # ── Left: Q(s) ───────────────────────────────────────────────────────
    ax1.plot(stimuli, Q, 'k-', lw=2, label="$Q(s)$")
    ax1.fill_between(stimuli, Q, where=Q > 0,
                     color='green', alpha=0.25, label="$Q>0$ (attractive)")
    ax1.fill_between(stimuli, Q, where=Q < 0,
                     color='red', alpha=0.25, label="$Q<0$ (repulsive)")
    ax1.axhline(0, color='k', lw=0.6, ls='--')
    ax1.set_xlabel("Stimulus $s$")
    ax1.set_ylabel("$Q(s)$")
    ax1.set_title("$Q(s) = \\frac{d}{ds}\\left["
                   "\\frac{p(s)}{J(s)^{(p+2)/4}}\\right]$")
    ax1.legend(fontsize=8)

    # ── Right: predicted vs. simulated bias ──────────────────────────────
    ax2.plot(stimuli, decomp['total_predicted'], 'b-', lw=2,
             label="Analytical total")
    if simulated_bias is not None:
        ax2.plot(stimuli, simulated_bias, 'ko', ms=4, alpha=0.6,
                 label="Simulated bias")
    ax2.axhline(0, color='k', lw=0.6, ls='--')
    # Color the background by sign of Q
    for i in range(len(stimuli) - 1):
        c = 'green' if Q[i] > 0 else 'red'
        ax2.axvspan(stimuli[i], stimuli[i + 1], alpha=0.06, color=c)
    ax2.set_xlabel("Stimulus $s$")
    ax2.set_ylabel("Bias")
    ax2.set_title("Bias with Q-sign background")
    ax2.legend(fontsize=8)

    fig.tight_layout()
    return fig


# =============================================================================
# Summary plot
# =============================================================================

def plot_summary(observer: BayesianObserver,
                 mlds_task: MLDSTask,
                 twoafc_task: TwoAFCTask,
                 psi_hat: np.ndarray,
                 stimuli: np.ndarray,
                 bias: np.ndarray,
                 threshold: np.ndarray,
                 predicted_bias: np.ndarray,
                 psi_hat_bayes: Optional[np.ndarray] = None) -> plt.Figure:
    """
    One-figure summary of the full pipeline (4 panels).

    Panels
    ------
    1. ψ recovery:        true ψ vs MLDS ψ̂ (and optional Bayesian MLDS ψ̂)
    2. Perceptual bias:   E[ŝ|s] − s
    3. Threshold:         std(ŝ|s) and analytical 1/√J(s)
    4. Wei & Stocker:     measured bias vs. d(threshold²)/dθ

    Parameters
    ----------
    observer : BayesianObserver
    mlds_task : MLDSTask
    twoafc_task : TwoAFCTask
    psi_hat : np.ndarray         — from MLDSTask.fit()
    stimuli : np.ndarray         — stimulus grid used for bias/threshold
    bias : np.ndarray            — from observer.compute_bias()
    threshold : np.ndarray       — from observer.compute_threshold()
    predicted_bias : np.ndarray  — np.gradient(threshold**2, stimuli)
    psi_hat_bayes : np.ndarray, optional — from Bayesian MLDS (Part 4)
    """
    S = observer.encoding.stimulus_space.get_grid()
    psi_true = observer.encoding.psi(S)
    # Analytical threshold from Fisher information
    s_fine = observer.encoding.stimulus_space.get_fine_grid()
    thresh_analytic = observer.encoding.discrimination_threshold(s_fine)

    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    fig.suptitle("Bayesian Perceptual Model — Full Pipeline", fontsize=13)

    # ---- Panel 1: ψ recovery ------------------------------------------------
    ax = axes[0]
    ax.plot(S, psi_true, 'k-', lw=2, label="True ψ")
    ax.plot(S, psi_hat, 'b--o', ms=5, label="MLDS ψ̂")
    if psi_hat_bayes is not None:
        ax.plot(S, psi_hat_bayes, 'r--s', ms=5, label="Bayesian MLDS ψ̂")
    ax.set_xlabel("Stimulus s")
    ax.set_ylabel("ψ(s)")
    ax.set_title("1. ψ Recovery (MLDS)")
    ax.legend(fontsize=8)

    # ---- Panel 2: Bias -------------------------------------------------------
    ax = axes[1]
    ax.plot(stimuli, bias, 'b-o', ms=5, label="Measured bias")
    ax.axhline(0, color='k', lw=0.8, ls='--')
    ax.set_xlabel("Stimulus s")
    ax.set_ylabel("Bias  E[ŝ|s] − s")
    ax.set_title("2. Perceptual Bias")
    ax.legend(fontsize=8)

    # ---- Panel 3: Threshold --------------------------------------------------
    ax = axes[2]
    ax.plot(stimuli, threshold, 'g-o', ms=5, label="Empirical threshold")
    ax.plot(s_fine, thresh_analytic, 'k--', lw=1.5, label="1/√J(s) (analytic)")
    ax.set_xlabel("Stimulus s")
    ax.set_ylabel("Threshold  std(ŝ|s)")
    ax.set_title("3. Discrimination Threshold")
    ax.legend(fontsize=8)

    # ---- Panel 4: Wei & Stocker law -----------------------------------------
    ax = axes[3]
    ax.plot(stimuli, bias, 'b-o', ms=5, label="Measured bias")
    ax.plot(stimuli, predicted_bias, 'r--o', ms=5, label="d(σ²)/dθ  (predicted)")
    ax.axhline(0, color='k', lw=0.8, ls='--')
    ax.set_xlabel("Stimulus s")
    ax.set_ylabel("Bias")
    ax.set_title("4. Wei & Stocker Law")
    ax.legend(fontsize=8)

    fig.tight_layout()
    return fig


# =============================================================================
# Demo — reproduces the 4-part notebook pipeline end-to-end
# =============================================================================

def run_demo(seed: int = 42,
             n_stim: int = 11,
             sigma: float = 0.1,
             n_trials_mlds: int = 300,
             n_trials_mlds_bayes: int = 5000,
             n_trials_2afc: int = 500,
             n_rep_bias: int = 500) -> None:
    """
    End-to-end demo reproducing the 4 parts of the original notebook.

    Part 1 — MLDS triplet task → recover ψ̂
    Part 2 — Bayesian observer → perceptual bias
    Part 3 — Verify Wei & Stocker law: bias ≈ d(threshold²)/dθ
    Part 4 — MLDS applied to a full Bayesian observer

    Parameters
    ----------
    seed : int
    n_stim : int          Number of stimulus levels (default: 11)
    sigma : float         Encoding noise std dev (default: 0.1)
    n_trials_mlds : int   Trials for Part 1 MLDS
    n_trials_mlds_bayes : int   Trials for Part 4 MLDS
    n_trials_2afc : int   Trials for TwoAFC (optional demo)
    n_rep_bias : int      Monte Carlo reps for bias/threshold estimation
    """
    rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    # Setup: shared objects for all 4 parts
    # ------------------------------------------------------------------
    space = StimulusSpace(0.0, 1.0, n_stim)

    psi_fn = make_psi('beta_cdf', space, a=10, b=3, mixture=True)
    enc    = EncodingModel(psi_fn, sigma=sigma, stimulus_space=space)
    prior  = Prior.beta_mixture(space, a=10, b=3, weight=0.5)
    # prior = make_prior('gaussian', space, mu=0.5, sigma=0.2)
    obs    = BayesianObserver(enc, prior, n_integration=400, rng=rng)

    S       = space.get_grid()
    s_fine  = space.get_fine_grid()
    stimuli = np.linspace(0.05, 0.95, n_stim)

    print("=" * 60)
    print("PART 1 — MLDS: Recovering ψ from triplet comparisons")
    print("=" * 60)

    mlds      = MLDSTask(obs, space, n_trials=n_trials_mlds,
                         sigma_decision=sigma,
                         use_bayesian_observer=False, rng=rng)
    data_mlds = mlds.run()
    psi_hat   = mlds.fit(data_mlds)
    psi_true  = enc.psi(S)

    fig1, ax = plt.subplots(figsize=(5, 4))
    ax.plot(S, psi_true, 'k-', lw=2, label="True ψ")
    ax.plot(S, psi_hat,  'b--o', ms=5, label=f"MLDS ψ̂  (n={n_trials_mlds})")
    ax.set_xlabel("Stimulus s")
    ax.set_ylabel("Perceptual scale ψ(s)")
    ax.set_title("Part 1 — ψ Recovery via MLDS")
    ax.legend()
    fig1.tight_layout()
    fig1.savefig("part1_mlds_recovery.png", dpi=120)
    print("  → part1_mlds_recovery.png")

    print()
    print("=" * 60)
    print("PART 2 — Bayesian inference: perceptual bias")
    print("=" * 60)

    bias_part2 = obs.compute_bias(stimuli, n_rep=n_rep_bias)

    # ── Prior & posterior figure ──────────────────────────────────────────────
    fig_pp = plot_prior_posterior(
        obs,
        example_stimuli=np.array([0.15, 0.3, 0.5, 0.7, 0.85]),
        noiseless=True,
        rng=rng,
    )
    fig_pp.savefig("part2_prior_posterior.png", dpi=120)
    print("  → part2_prior_posterior.png")

    # ── Bias figure ───────────────────────────────────────────────────────────
    fig2, ax = plt.subplots(figsize=(5, 4))
    ax.plot(stimuli, bias_part2, 'b-o', ms=5)
    ax.axhline(0, color='k', lw=0.8, ls='--')
    ax.set_xlabel("True stimulus s")
    ax.set_ylabel("Bias  E[ŝ|s] − s")
    ax.set_title("Part 2 — Bayesian Perceptual Bias")
    fig2.tight_layout()
    fig2.savefig("part2_bias.png", dpi=120)
    print("  → part2_bias.png")

    print()
    print("=" * 60)
    print("PART 3 — Wei & Stocker law verification")
    print("=" * 60)

    bias_ws, predicted_bias = obs.check_wei_stocker_law(
        stimuli, n_rep=n_rep_bias)
    # Also collect threshold separately for the summary figure
    _, threshold = obs._collect_estimates(stimuli, n_rep=n_rep_bias, loss='L2')

    fig3, ax = plt.subplots(figsize=(5, 4))
    ax.plot(stimuli, bias_ws,       'b-o',  ms=5, label="Measured bias")
    ax.plot(stimuli, predicted_bias, 'r--o', ms=5, label="d(σ²)/dθ  (predicted)")
    ax.axhline(0, color='k', lw=0.8, ls='--')
    ax.set_xlabel("Stimulus s")
    ax.set_ylabel("Bias")
    ax.set_title("Part 3 — Wei & Stocker Law")
    ax.legend()
    fig3.tight_layout()
    fig3.savefig("part3_wei_stocker.png", dpi=120)
    print("  → part3_wei_stocker.png")

    print()
    print("=" * 60)
    print("PART 4 — MLDS on the Bayesian observer")
    print("=" * 60)

    mlds_bayes      = MLDSTask(obs, space, n_trials=n_trials_mlds_bayes,
                               sigma_decision=sigma,
                               use_bayesian_observer=True, rng=rng)
    data_bayes      = mlds_bayes.run()
    psi_hat_bayes   = mlds_bayes.fit(data_bayes)

    # The Bayesian MLDS should recover the posterior-mean curve, not the raw ψ.
    # We overlay the rescaled posterior mean for comparison.
    pm = np.array([obs.simulate_estimate(s) for s in S])
    pm_rescaled = (pm - pm[0]) / (pm[-1] - pm[0] + 1e-10)

    fig4, ax = plt.subplots(figsize=(5, 4))
    ax.plot(S, psi_true,      'k-',   lw=2,  label="True ψ")
    ax.plot(S, pm_rescaled,   'g-',   lw=1.5, label="Posterior mean (rescaled)")
    ax.plot(S, psi_hat_bayes, 'r--o', ms=5,   label=f"Bayesian MLDS ψ̂  (n={n_trials_mlds_bayes})")
    ax.set_xlabel("Stimulus s")
    ax.set_ylabel("Perceptual scale")
    ax.set_title("Part 4 — MLDS on Bayesian Observer")
    ax.legend()
    fig4.tight_layout()
    fig4.savefig("part4_bayes_mlds.png", dpi=120)
    print("  → part4_bayes_mlds.png")

    print()
    print("=" * 60)
    print("PART 5 — Hahn & Wei (2024) bias decomposition")
    print("=" * 60)

    # Use a finer grid for analytical curves (more points → smoother gradients)
    stimuli_fine = np.linspace(0.05, 0.95, 60)

    # Simulated bias on the fine grid (re-use same observer)
    bias_sim_fine = obs.compute_bias(stimuli_fine, n_rep=n_rep_bias)

    # ── 5a: Decomposition figure (Prior | Resources | Attraction | Repulsion | Total)
    fig5a = plot_hahn_decomposition(
        obs,
        stimuli_fine,
        simulated_bias=bias_sim_fine,
        p_values=[0, 1, 2, 4, 8],
    )
    fig5a.savefig("part5a_hahn_decomposition.png", dpi=120)
    print("  → part5a_hahn_decomposition.png")

    # ── 5b: P/P ratio rule
    fig5b = plot_pp_ratio(
        obs,
        stimuli_fine,
        simulated_bias=bias_sim_fine,
        p=2.0,
    )
    fig5b.savefig("part5b_pp_ratio.png", dpi=120)
    print("  → part5b_pp_ratio.png")

    # ── 5c: Compare analytical (Hahn) vs. Wei & Stocker (repulsion-only)
    decomp_L2 = obs.decompose_bias(stimuli_fine, p=2.0)

    fig5c, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(stimuli_fine, bias_sim_fine, 'ko', ms=4, alpha=0.5,
            label="Simulated bias (MC)")
    ax.plot(stimuli_fine, decomp_L2['total_predicted'], 'b-', lw=2,
            label="Hahn total  (attr. + rep.)")
    ax.plot(stimuli_fine, decomp_L2['attraction'], 'g--', lw=1.5,
            label="Prior attraction")
    ax.plot(stimuli_fine, decomp_L2['repulsion'], 'r--', lw=1.5,
            label="Likelihood repulsion")
    # Wei & Stocker (repulsion-only) prediction for comparison
    ws_pred = np.gradient(1.0 / decomp_L2['J'], stimuli_fine)
    ax.plot(stimuli_fine, ws_pred, 'm:', lw=1.5,
            label="W&S 2017  d(1/J)/ds  (rep. only)")
    ax.axhline(0, color='k', lw=0.6, ls='--')
    ax.set_xlabel("Stimulus $s$")
    ax.set_ylabel("Bias")
    ax.set_title("Part 5c — Hahn vs. Wei & Stocker: why repulsion alone is not enough")
    ax.legend(fontsize=8)
    fig5c.tight_layout()
    fig5c.savefig("part5c_hahn_vs_ws.png", dpi=120)
    print("  → part5c_hahn_vs_ws.png")

    print()
    print("=" * 60)
    print("SUMMARY FIGURE (all 4 panels)")
    print("=" * 60)

    twoafc = TwoAFCTask(obs, space, n_trials=n_trials_2afc, rng=rng)

    fig_summary = plot_summary(
        observer=obs,
        mlds_task=mlds,
        twoafc_task=twoafc,
        psi_hat=psi_hat,
        stimuli=stimuli,
        bias=bias_ws,
        threshold=threshold,
        predicted_bias=predicted_bias,
        psi_hat_bayes=psi_hat_bayes,
    )
    fig_summary.savefig("summary.png", dpi=120)
    print("  → summary.png")

    plt.show()
    print("\nDone.")


if __name__ == "__main__":
    run_demo()
