"""
Risk analytics engine.

Implements:
  - Historical simulation VaR / CVaR (Expected Shortfall)
  - Parametric VaR (normal and Student-t)
  - Cornish-Fisher modified VaR (accounts for skewness & kurtosis)
  - Monte Carlo VaR with variance reduction
  - Maximum drawdown and underwater analysis
  - Stress testing (historical scenarios + custom shocks)
  - Marginal and component VaR decomposition
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import norm, t as student_t
from typing import Dict, List, Optional, Tuple


class RiskEngine:
    """
    Comprehensive risk analytics for a portfolio or single return series.

    Parameters
    ----------
    returns : pd.Series or pd.DataFrame
        Asset or portfolio returns (daily).
    weights : pd.Series, optional
        Portfolio weights (required if returns is a DataFrame).
    trading_days : int
        Trading days per year (default 252).
    """

    def __init__(
        self,
        returns: pd.Series | pd.DataFrame,
        weights: Optional[pd.Series] = None,
        trading_days: int = 252,
    ):
        self.trading_days = trading_days

        if isinstance(returns, pd.DataFrame):
            if weights is None:
                weights = pd.Series(
                    np.ones(returns.shape[1]) / returns.shape[1],
                    index=returns.columns,
                )
            self.asset_returns  = returns.dropna()
            self.weights        = weights.reindex(returns.columns).fillna(0)
            self.port_returns   = (self.asset_returns * self.weights).sum(axis=1)
        else:
            self.asset_returns  = None
            self.weights        = None
            self.port_returns   = returns.dropna()

    # ------------------------------------------------------------------
    # Value at Risk
    # ------------------------------------------------------------------

    def historical_var(self, confidence: float = 0.95, horizon: int = 1) -> float:
        """
        Historical simulation VaR.

        Parameters
        ----------
        confidence : float
            Confidence level (e.g. 0.95 = 95% VaR).
        horizon : int
            Horizon in trading days.

        Returns
        -------
        float : VaR as a positive number (loss magnitude).
        """
        r = self.port_returns
        scaled = r * np.sqrt(horizon)   # scale for horizon (assumes i.i.d.)
        return float(-np.percentile(scaled.dropna(), (1 - confidence) * 100))

    def parametric_var(
        self,
        confidence: float = 0.95,
        horizon: int = 1,
        dist: str = "normal",
        dof: int = 5,
    ) -> float:
        """
        Parametric VaR assuming normal or Student-t distribution.

        Parameters
        ----------
        dist : 'normal' or 't'
        dof  : degrees of freedom for Student-t (heavier tails).
        """
        mu    = self.port_returns.mean()
        sigma = self.port_returns.std()

        if dist == "normal":
            z = norm.ppf(1 - confidence)
        elif dist == "t":
            z = student_t.ppf(1 - confidence, df=dof)
        else:
            raise ValueError("dist must be 'normal' or 't'")

        daily_var = -(mu + z * sigma)
        return float(daily_var * np.sqrt(horizon))

    def cornish_fisher_var(
        self,
        confidence: float = 0.95,
        horizon: int = 1,
    ) -> float:
        """
        Cornish-Fisher modified VaR.

        Adjusts the normal quantile for the observed skewness (S)
        and excess kurtosis (K) of the return distribution:

            z_CF = z + (z²-1)*S/6 + (z³-3z)*K/24 - (2z³-5z)*S²/36

        More accurate than parametric VaR for fat-tailed distributions.
        """
        r  = self.port_returns.dropna()
        mu = r.mean()
        sigma = r.std()
        S = float(stats.skew(r))
        K = float(stats.kurtosis(r))   # excess kurtosis

        z = norm.ppf(1 - confidence)
        z_cf = (
            z
            + (z**2 - 1) * S / 6
            + (z**3 - 3 * z) * K / 24
            - (2 * z**3 - 5 * z) * S**2 / 36
        )

        daily_var = -(mu + z_cf * sigma)
        return float(daily_var * np.sqrt(horizon))

    def monte_carlo_var(
        self,
        confidence: float = 0.95,
        horizon: int = 1,
        n_sims: int = 100_000,
        seed: int = 42,
    ) -> float:
        """
        Monte Carlo VaR using antithetic variates for variance reduction.

        Simulates horizon-day cumulative returns from a multivariate
        normal distribution fitted to observed returns.
        """
        rng = np.random.default_rng(seed)
        mu    = float(self.port_returns.mean())
        sigma = float(self.port_returns.std())

        half = n_sims // 2
        z = rng.standard_normal(half * horizon).reshape(half, horizon)
        z = np.vstack([z, -z])          # antithetic pairs

        daily = mu + sigma * z
        cumulative = daily.sum(axis=1)

        return float(-np.percentile(cumulative, (1 - confidence) * 100))

    # ------------------------------------------------------------------
    # Expected Shortfall (CVaR)
    # ------------------------------------------------------------------

    def cvar(self, confidence: float = 0.95, horizon: int = 1) -> float:
        """
        Conditional Value at Risk (Expected Shortfall).

        Average loss in the worst (1 - confidence) fraction of outcomes.
        Always >= VaR; gives information about tail severity.
        """
        r = self.port_returns * np.sqrt(horizon)
        threshold = np.percentile(r, (1 - confidence) * 100)
        tail = r[r <= threshold]
        return float(-tail.mean())

    # ------------------------------------------------------------------
    # Drawdown analysis
    # ------------------------------------------------------------------

    def drawdown_series(self, returns: Optional[pd.Series] = None) -> pd.Series:
        """Return the drawdown series (fraction below running peak)."""
        r = returns if returns is not None else self.port_returns
        wealth = (1 + r).cumprod()
        peak   = wealth.cummax()
        return (wealth - peak) / peak

    def max_drawdown(self, returns: Optional[pd.Series] = None) -> float:
        """Maximum peak-to-trough drawdown (positive number)."""
        return float(-self.drawdown_series(returns).min())

    def drawdown_stats(self) -> Dict:
        """
        Full drawdown statistics:
          max_drawdown, max_duration (days), avg_drawdown, current_drawdown
        """
        dd = self.drawdown_series()
        in_dd = dd < 0
        durations = []
        current_len = 0

        for is_down in in_dd:
            if is_down:
                current_len += 1
            else:
                if current_len > 0:
                    durations.append(current_len)
                current_len = 0
        if current_len > 0:
            durations.append(current_len)

        return {
            "max_drawdown":     round(-dd.min(), 4),
            "max_duration_days": max(durations) if durations else 0,
            "avg_drawdown":     round(-dd[dd < 0].mean(), 4) if (dd < 0).any() else 0,
            "current_drawdown": round(float(dd.iloc[-1]), 4),
        }

    # ------------------------------------------------------------------
    # Stress testing
    # ------------------------------------------------------------------

    HISTORICAL_SCENARIOS = {
        "GFC_2008":          {"equity": -0.40, "bonds": 0.05,  "commodities": -0.30},
        "COVID_Crash_2020":  {"equity": -0.34, "bonds": 0.08,  "commodities": -0.25},
        "DotCom_2000":       {"equity": -0.45, "bonds": 0.10,  "commodities": 0.05},
        "Black_Monday_1987": {"equity": -0.22, "bonds": 0.02,  "commodities": -0.05},
        "Rate_Shock_2022":   {"equity": -0.25, "bonds": -0.18, "commodities": 0.30},
        "Taper_Tantrum_2013":{"equity": -0.05, "bonds": -0.08, "commodities": -0.10},
    }

    def stress_test(
        self,
        factor_exposures: Dict[str, float],
        custom_scenarios: Optional[Dict[str, Dict[str, float]]] = None,
    ) -> pd.DataFrame:
        """
        Estimate portfolio P&L under named stress scenarios.

        Parameters
        ----------
        factor_exposures : dict
            Portfolio's sensitivity to each risk factor.
            e.g. {"equity": 0.8, "bonds": 0.1, "commodities": 0.05}
        custom_scenarios : dict, optional
            Additional user-defined scenarios.

        Returns
        -------
        pd.DataFrame with scenario name and estimated P&L.
        """
        scenarios = dict(self.HISTORICAL_SCENARIOS)
        if custom_scenarios:
            scenarios.update(custom_scenarios)

        results = []
        for name, shocks in scenarios.items():
            pnl = sum(
                factor_exposures.get(factor, 0) * shock
                for factor, shock in shocks.items()
            )
            results.append({"scenario": name, "estimated_pnl": round(pnl, 4)})

        df = pd.DataFrame(results).set_index("scenario")
        return df.sort_values("estimated_pnl")

    # ------------------------------------------------------------------
    # Component & Marginal VaR
    # ------------------------------------------------------------------

    def component_var(
        self,
        confidence: float = 0.95,
    ) -> Optional[pd.Series]:
        """
        Component VaR: each asset's contribution to portfolio VaR.

        Requires asset_returns and weights to be set.
        Component VaR sums exactly to total portfolio VaR.
        """
        if self.asset_returns is None:
            return None

        cov   = self.asset_returns.cov().values * self.trading_days
        w     = self.weights.values
        sigma = np.sqrt(w @ cov @ w)
        z     = norm.ppf(confidence)

        marginal_var = z * (cov @ w) / (sigma + 1e-12)
        component    = w * marginal_var

        return pd.Series(component, index=self.asset_returns.columns)

    # ------------------------------------------------------------------
    # Full risk report
    # ------------------------------------------------------------------

    def full_report(self, confidence: float = 0.95) -> Dict:
        """Return a dictionary of all key risk metrics."""
        return {
            "historical_var_95":     round(self.historical_var(0.95), 4),
            "historical_var_99":     round(self.historical_var(0.99), 4),
            "parametric_var_normal": round(self.parametric_var(confidence), 4),
            "parametric_var_t":      round(self.parametric_var(confidence, dist="t"), 4),
            "cornish_fisher_var":    round(self.cornish_fisher_var(confidence), 4),
            "monte_carlo_var":       round(self.monte_carlo_var(confidence), 4),
            "cvar_95":               round(self.cvar(0.95), 4),
            "cvar_99":               round(self.cvar(0.99), 4),
            "max_drawdown":          round(self.max_drawdown(), 4),
            "skewness":              round(float(stats.skew(self.port_returns)), 4),
            "excess_kurtosis":       round(float(stats.kurtosis(self.port_returns)), 4),
            **self.drawdown_stats(),
        }
