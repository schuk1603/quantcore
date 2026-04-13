"""
Market regime detection.

Implements:
  - Hidden Markov Model (HMM) regime detection (bull/bear/sideways)
  - Volatility regime classification
  - Trend regime via rolling linear regression slope
  - Regime-conditional statistics

Inspired by academic research on regime-switching models
(Hamilton 1989, Ang & Bekaert 2002). Original implementation.
"""

import numpy as np
import pandas as pd
from scipy.stats import norm
from typing import Dict, List, Tuple


class RegimeDetector:
    """
    Gaussian Hidden Markov Model for market regime detection.

    Fits a K-state HMM to return data using the Baum-Welch (EM) algorithm.
    States are labelled by their mean return (bull/bear/sideways).

    Parameters
    ----------
    n_states : int
        Number of hidden states (default 3: bull, sideways, bear).
    max_iter : int
        Maximum EM iterations.
    tol : float
        Convergence tolerance.
    """

    def __init__(self, n_states: int = 3, max_iter: int = 200, tol: float = 1e-4):
        self.n_states = n_states
        self.max_iter = max_iter
        self.tol      = tol

        # Model parameters (fitted)
        self.pi    = None   # initial state distribution
        self.A     = None   # transition matrix [n_states x n_states]
        self.mu    = None   # state means
        self.sigma = None   # state std devs

    # ------------------------------------------------------------------
    # Fit via Baum-Welch EM
    # ------------------------------------------------------------------

    def fit(self, returns: pd.Series) -> "RegimeDetector":
        """Fit HMM parameters to a return series via Baum-Welch EM."""
        r = returns.dropna().values
        K = self.n_states
        T = len(r)

        # Initialise parameters via k-means-like split
        sorted_r = np.sort(r)
        splits   = np.array_split(sorted_r, K)
        self.mu    = np.array([s.mean() for s in splits])
        self.sigma = np.array([max(s.std(), 1e-4) for s in splits])
        self.pi    = np.ones(K) / K
        self.A     = np.ones((K, K)) / K

        prev_ll = -np.inf

        for _ in range(self.max_iter):
            # E-step: forward-backward
            alpha, scale = self._forward(r)
            beta         = self._backward(r, scale)
            gamma, xi    = self._gamma_xi(r, alpha, beta)

            # M-step
            self.pi    = gamma[0] / gamma[0].sum()
            self.A     = xi.sum(0) / xi.sum(0).sum(1, keepdims=True)
            self.mu    = (gamma * r[:, None]).sum(0) / gamma.sum(0)
            var        = (gamma * (r[:, None] - self.mu) ** 2).sum(0) / gamma.sum(0)
            self.sigma = np.sqrt(np.maximum(var, 1e-8))

            ll = np.log(scale + 1e-300).sum()
            if abs(ll - prev_ll) < self.tol:
                break
            prev_ll = ll

        # Label states: sort by mean return (0=bear, 1=sideways, 2=bull)
        order      = np.argsort(self.mu)
        self.mu    = self.mu[order]
        self.sigma = self.sigma[order]
        self.pi    = self.pi[order]
        self.A     = self.A[np.ix_(order, order)]

        return self

    def predict(self, returns: pd.Series) -> pd.Series:
        """
        Return the most likely state sequence (Viterbi decoding).

        States: 0=bear, 1=sideways, 2=bull
        """
        r = returns.dropna().values
        states = self._viterbi(r)
        labels = pd.Series(states, index=returns.dropna().index, name="regime")
        return labels

    def regime_probs(self, returns: pd.Series) -> pd.DataFrame:
        """
        Return smoothed posterior state probabilities at each time step.
        Columns: bear, sideways, bull
        """
        r = returns.dropna().values
        alpha, scale = self._forward(r)
        beta         = self._backward(r, scale)
        gamma, _     = self._gamma_xi(r, alpha, beta)

        names = ["bear", "sideways", "bull"][: self.n_states]
        return pd.DataFrame(gamma, index=returns.dropna().index, columns=names)

    def regime_stats(
        self, returns: pd.Series, regime_labels: pd.Series
    ) -> pd.DataFrame:
        """
        Compute annualised return, volatility, and Sharpe per regime.
        """
        combined = pd.concat([returns, regime_labels], axis=1, join="inner")
        combined.columns = ["returns", "regime"]
        state_names = {0: "Bear", 1: "Sideways", 2: "Bull"}
        records = []
        for state in sorted(combined["regime"].unique()):
            r = combined.loc[combined["regime"] == state, "returns"]
            records.append({
                "regime":     state_names.get(state, str(state)),
                "mean_ret":   round(r.mean() * 252, 4),
                "volatility": round(r.std() * np.sqrt(252), 4),
                "sharpe":     round(r.mean() / (r.std() + 1e-8) * np.sqrt(252), 4),
                "pct_time":   round(len(r) / len(combined), 4),
                "n_days":     len(r),
            })
        return pd.DataFrame(records).set_index("regime")

    # ------------------------------------------------------------------
    # Forward-backward algorithm
    # ------------------------------------------------------------------

    def _emission(self, r: np.ndarray) -> np.ndarray:
        """Gaussian emission probabilities [T x K]."""
        return np.column_stack([
            norm.pdf(r, self.mu[k], self.sigma[k])
            for k in range(self.n_states)
        ])

    def _forward(self, r: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        T, K = len(r), self.n_states
        alpha = np.zeros((T, K))
        scale = np.zeros(T)
        B     = self._emission(r)

        alpha[0] = self.pi * B[0]
        scale[0] = alpha[0].sum()
        alpha[0] /= scale[0] + 1e-300

        for t in range(1, T):
            alpha[t] = (alpha[t - 1] @ self.A) * B[t]
            scale[t] = alpha[t].sum()
            alpha[t] /= scale[t] + 1e-300

        return alpha, scale

    def _backward(self, r: np.ndarray, scale: np.ndarray) -> np.ndarray:
        T, K = len(r), self.n_states
        beta  = np.zeros((T, K))
        B     = self._emission(r)
        beta[-1] = 1.0

        for t in range(T - 2, -1, -1):
            beta[t] = (self.A * B[t + 1] * beta[t + 1]).sum(1)
            beta[t] /= scale[t + 1] + 1e-300

        return beta

    def _gamma_xi(
        self, r: np.ndarray, alpha: np.ndarray, beta: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        T, K = len(r), self.n_states
        B     = self._emission(r)
        gamma = alpha * beta
        gamma /= gamma.sum(1, keepdims=True) + 1e-300

        xi = np.zeros((T - 1, K, K))
        for t in range(T - 1):
            xi[t] = (
                alpha[t][:, None]
                * self.A
                * B[t + 1]
                * beta[t + 1]
            )
            xi[t] /= xi[t].sum() + 1e-300

        return gamma, xi

    def _viterbi(self, r: np.ndarray) -> np.ndarray:
        T, K = len(r), self.n_states
        B    = self._emission(r)
        log_A  = np.log(self.A  + 1e-300)
        log_pi = np.log(self.pi + 1e-300)
        log_B  = np.log(B       + 1e-300)

        delta = np.zeros((T, K))
        psi   = np.zeros((T, K), dtype=int)

        delta[0] = log_pi + log_B[0]
        for t in range(1, T):
            trans = delta[t - 1][:, None] + log_A
            psi[t]   = trans.argmax(0)
            delta[t] = trans.max(0) + log_B[t]

        states    = np.zeros(T, dtype=int)
        states[-1] = delta[-1].argmax()
        for t in range(T - 2, -1, -1):
            states[t] = psi[t + 1, states[t + 1]]

        return states
