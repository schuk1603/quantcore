"""
Portfolio optimization.

Implements:
  - Mean-Variance Optimization (Markowitz 1952)
  - Maximum Sharpe Ratio portfolio
  - Minimum Variance portfolio
  - Black-Litterman model (He & Litterman 1999)
  - Hierarchical Risk Parity (Lopez de Prado 2016)
  - Risk Parity / Equal Risk Contribution
"""

import numpy as np
import pandas as pd
import cvxpy as cp
from scipy.optimize import minimize
from scipy.cluster.hierarchy import linkage, dendrogram, leaves_list
from scipy.spatial.distance import squareform
from typing import Dict, List, Optional, Tuple


class PortfolioOptimizer:
    """
    Multi-method portfolio optimizer.

    Parameters
    ----------
    returns : pd.DataFrame
        Asset returns (rows = dates, columns = assets).
    risk_free_rate : float
        Annualized risk-free rate (default 0.0).
    trading_days : int
        Trading days per year (default 252).
    """

    def __init__(
        self,
        returns: pd.DataFrame,
        risk_free_rate: float = 0.0,
        trading_days: int = 252,
    ):
        self.returns       = returns.dropna()
        self.risk_free_rate = risk_free_rate
        self.trading_days  = trading_days
        self.n             = returns.shape[1]
        self.tickers       = list(returns.columns)

        self.mu    = returns.mean() * trading_days
        self.sigma = returns.cov()  * trading_days

    # ------------------------------------------------------------------
    # Mean-Variance Optimization (Markowitz)
    # ------------------------------------------------------------------

    def mean_variance(
        self,
        target_return: Optional[float] = None,
        long_only: bool = True,
        max_weight: float = 1.0,
    ) -> pd.Series:
        """
        Minimum variance portfolio for a given target return,
        or the global minimum variance if no target is provided.

        Parameters
        ----------
        target_return : float, optional
            Annualized target portfolio return.
        long_only : bool
            Constrain weights to be non-negative.
        max_weight : float
            Maximum weight per asset.

        Returns
        -------
        pd.Series of optimal weights indexed by ticker.
        """
        w = cp.Variable(self.n)
        sigma_np = self.sigma.values

        objective = cp.Minimize(cp.quad_form(w, sigma_np))

        constraints = [cp.sum(w) == 1]
        if long_only:
            constraints.append(w >= 0)
        constraints.append(w <= max_weight)

        if target_return is not None:
            constraints.append(self.mu.values @ w >= target_return)

        prob = cp.Problem(objective, constraints)
        prob.solve(solver=cp.CLARABEL, warm_start=True)

        if prob.status not in ("optimal", "optimal_inaccurate"):
            raise RuntimeError(f"Optimization failed: {prob.status}")

        return pd.Series(w.value, index=self.tickers)

    # ------------------------------------------------------------------
    # Maximum Sharpe Ratio
    # ------------------------------------------------------------------

    def max_sharpe(
        self,
        long_only: bool = True,
        max_weight: float = 1.0,
    ) -> pd.Series:
        """
        Maximise the Sharpe ratio using the Sharpe-ratio maximisation
        trick (Cornuejols & Tutuncu, 2006): transform to a convex QP
        by rescaling weights.
        """
        rf_daily = self.risk_free_rate
        mu_excess = (self.mu - rf_daily).values
        sigma_np  = self.sigma.values

        # Auxiliary variable y = w / (mu_excess @ w)
        y = cp.Variable(self.n)
        kappa = cp.Variable()   # kappa = 1 / (mu_excess @ w)

        objective = cp.Minimize(cp.quad_form(y, sigma_np))
        constraints = [mu_excess @ y == 1, kappa >= 0, cp.sum(y) == kappa]
        if long_only:
            constraints.append(y >= 0)
        constraints.append(y <= max_weight * kappa)

        prob = cp.Problem(objective, constraints)
        prob.solve(solver=cp.CLARABEL)

        if prob.status not in ("optimal", "optimal_inaccurate") or kappa.value is None:
            # Fallback to scipy
            return self._max_sharpe_scipy(long_only, max_weight)

        w = y.value / (kappa.value + 1e-12)
        return pd.Series(w / w.sum(), index=self.tickers)

    def _max_sharpe_scipy(self, long_only: bool, max_weight: float) -> pd.Series:
        mu  = self.mu.values
        cov = self.sigma.values
        rf  = self.risk_free_rate

        def neg_sharpe(w):
            port_ret = w @ mu
            port_std = np.sqrt(w @ cov @ w + 1e-10)
            return -(port_ret - rf) / port_std

        bounds = [(0, max_weight)] * self.n if long_only else [(-1, max_weight)] * self.n
        constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1}
        w0 = np.ones(self.n) / self.n

        result = minimize(neg_sharpe, w0, method="SLSQP",
                          bounds=bounds, constraints=constraints,
                          options={"maxiter": 1000, "ftol": 1e-9})

        return pd.Series(result.x, index=self.tickers)

    # ------------------------------------------------------------------
    # Black-Litterman
    # ------------------------------------------------------------------

    def black_litterman(
        self,
        market_caps: pd.Series,
        views: List[Dict],
        tau: float = 0.05,
        risk_aversion: float = 2.5,
    ) -> Tuple[pd.Series, pd.Series]:
        """
        Black-Litterman model.

        Parameters
        ----------
        market_caps : pd.Series
            Market capitalisations indexed by ticker.
        views : list of dicts, each with keys:
            'assets'  : list of tickers (positive) + list (negative, optional)
            'Q'       : expected return of the view (annualised)
            'omega'   : confidence / uncertainty (variance of view error)
            Example:
              {'assets': ['AAPL'], 'Q': 0.10, 'omega': 0.02}  # AAPL +10%
              {'assets': ['AAPL', 'MSFT'], 'positions': [1, -1],
               'Q': 0.05, 'omega': 0.01}                       # AAPL - MSFT = +5%
        tau : float
            Scaling factor for prior uncertainty (typically 0.01–0.10).
        risk_aversion : float
            Market risk-aversion coefficient (typically 2–4).

        Returns
        -------
        mu_bl    : posterior expected returns
        w_bl     : Black-Litterman optimal weights
        """
        # Market-cap weights
        w_mkt = market_caps.reindex(self.tickers).fillna(0)
        w_mkt = w_mkt / w_mkt.sum()

        Sigma = self.sigma.values
        # Implied equilibrium excess returns
        Pi = risk_aversion * Sigma @ w_mkt.values

        if not views:
            mu_bl = pd.Series(Pi, index=self.tickers)
            w_bl  = self.max_sharpe()
            return mu_bl, w_bl

        # Build P (view) matrix and Q (view return) vector
        k = len(views)
        P = np.zeros((k, self.n))
        Q = np.zeros(k)
        Omega_diag = np.zeros(k)

        for i, view in enumerate(views):
            assets    = view.get("assets", [])
            positions = view.get("positions", None)
            Q[i]      = view["Q"]
            Omega_diag[i] = view["omega"]

            if positions is not None:
                for asset, pos in zip(assets, positions):
                    if asset in self.tickers:
                        P[i, self.tickers.index(asset)] = pos
            else:
                if len(assets) == 1 and assets[0] in self.tickers:
                    P[i, self.tickers.index(assets[0])] = 1.0

        Omega = np.diag(Omega_diag)

        # Posterior return = BL formula
        tSigma_inv = np.linalg.inv(tau * Sigma)
        Pt_Omega_inv_P = P.T @ np.linalg.inv(Omega) @ P

        posterior_cov = np.linalg.inv(tSigma_inv + Pt_Omega_inv_P)
        posterior_mu  = posterior_cov @ (tSigma_inv @ Pi + P.T @ np.linalg.inv(Omega) @ Q)

        mu_bl = pd.Series(posterior_mu, index=self.tickers)
        Sigma_bl = Sigma + posterior_cov

        # Optimal weights from posterior
        opt = PortfolioOptimizer.__new__(PortfolioOptimizer)
        opt.n       = self.n
        opt.tickers = self.tickers
        opt.mu      = mu_bl
        opt.sigma   = pd.DataFrame(Sigma_bl, index=self.tickers, columns=self.tickers)
        opt.risk_free_rate = self.risk_free_rate
        opt.returns = self.returns

        try:
            w_bl = opt.max_sharpe()
        except Exception:
            w_bl = pd.Series(w_mkt.values, index=self.tickers)

        return mu_bl, w_bl

    # ------------------------------------------------------------------
    # Hierarchical Risk Parity (Lopez de Prado 2016)
    # ------------------------------------------------------------------

    def hierarchical_risk_parity(self) -> pd.Series:
        """
        Hierarchical Risk Parity (HRP).

        Steps:
          1. Compute correlation matrix and convert to distance matrix.
          2. Hierarchical clustering (Ward linkage).
          3. Quasi-diagonalise the covariance matrix.
          4. Recursive bisection to allocate weights.

        Returns
        -------
        pd.Series of HRP weights.
        """
        corr = self.returns.corr()
        cov  = self.sigma

        # Distance matrix: d_ij = sqrt(0.5 * (1 - rho_ij))
        dist = np.sqrt(0.5 * (1 - corr.values))
        np.fill_diagonal(dist, 0)

        # Hierarchical clustering
        condensed = squareform(dist, checks=False)
        link      = linkage(condensed, method="ward")
        order     = leaves_list(link)

        # Quasi-diagonal sort
        sorted_tickers = [self.tickers[i] for i in order]
        sorted_cov     = cov.loc[sorted_tickers, sorted_tickers]

        weights = self._recursive_bisection(sorted_cov, sorted_tickers)
        return weights.reindex(self.tickers).fillna(0)

    def _recursive_bisection(
        self,
        cov: pd.DataFrame,
        tickers: List[str],
    ) -> pd.Series:
        weights = pd.Series(1.0, index=tickers)
        clusters = [tickers]

        while clusters:
            clusters = [
                c[start:end]
                for c in clusters
                for start, end in [(0, len(c) // 2), (len(c) // 2, len(c))]
                if len(c) > 1
            ]

            for i in range(0, len(clusters), 2):
                if i + 1 >= len(clusters):
                    break
                c1 = clusters[i]
                c2 = clusters[i + 1]

                v1 = self._cluster_var(cov, c1)
                v2 = self._cluster_var(cov, c2)
                alpha = 1 - v1 / (v1 + v2 + 1e-12)

                weights[c1] *= alpha
                weights[c2] *= (1 - alpha)

        return weights / weights.sum()

    @staticmethod
    def _cluster_var(cov: pd.DataFrame, cluster: List[str]) -> float:
        sub_cov = cov.loc[cluster, cluster].values
        inv_diag = 1.0 / np.diag(sub_cov)
        w = inv_diag / inv_diag.sum()
        return float(w @ sub_cov @ w)

    # ------------------------------------------------------------------
    # Risk Parity (Equal Risk Contribution)
    # ------------------------------------------------------------------

    def risk_parity(self, max_iter: int = 1000) -> pd.Series:
        """
        Equal Risk Contribution portfolio.

        Each asset contributes the same amount to total portfolio risk.
        Solved via convex optimisation (Spinu 2013).
        """
        cov = self.sigma.values
        n   = self.n

        def portfolio_risk(w):
            return np.sqrt(w @ cov @ w)

        def risk_contributions(w):
            pr   = portfolio_risk(w)
            mrc  = cov @ w / (pr + 1e-12)
            return w * mrc

        def objective(w):
            rc  = risk_contributions(w)
            avg = rc.mean()
            return np.sum((rc - avg) ** 2)

        bounds = [(1e-4, 1.0)] * n
        constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1}
        w0 = np.ones(n) / n

        result = minimize(
            objective, w0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": max_iter, "ftol": 1e-12},
        )

        return pd.Series(result.x / result.x.sum(), index=self.tickers)

    # ------------------------------------------------------------------
    # Efficient Frontier
    # ------------------------------------------------------------------

    def efficient_frontier(
        self,
        n_points: int = 50,
        long_only: bool = True,
    ) -> pd.DataFrame:
        """
        Compute the efficient frontier.

        Returns a DataFrame with columns:
          return, volatility, sharpe, weights_<ticker>
        """
        min_ret = float(self.mu.min())
        max_ret = float(self.mu.max())
        target_returns = np.linspace(min_ret, max_ret, n_points)

        records = []
        for tr in target_returns:
            try:
                w = self.mean_variance(target_return=tr, long_only=long_only)
                port_ret = float(self.mu @ w)
                port_vol = float(np.sqrt(w @ self.sigma @ w))
                sharpe   = (port_ret - self.risk_free_rate) / (port_vol + 1e-8)
                row = {"return": port_ret, "volatility": port_vol, "sharpe": sharpe}
                row.update({f"w_{t}": float(w[t]) for t in self.tickers})
                records.append(row)
            except Exception:
                continue

        return pd.DataFrame(records)
