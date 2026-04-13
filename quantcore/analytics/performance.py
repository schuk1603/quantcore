"""
Performance analytics and reporting.

Implements the full suite of institutional-grade performance metrics.
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, Optional


class PerformanceAnalytics:
    """
    Compute risk-adjusted performance metrics from a return series.

    Parameters
    ----------
    returns : pd.Series
        Daily portfolio returns.
    benchmark : pd.Series, optional
        Daily benchmark returns for relative metrics.
    risk_free_rate : float
        Annualized risk-free rate (default 0.0).
    trading_days : int
        Trading days per year (default 252).
    """

    def __init__(
        self,
        returns: pd.Series,
        benchmark: Optional[pd.Series] = None,
        risk_free_rate: float = 0.0,
        trading_days: int = 252,
    ):
        self.returns       = returns.dropna()
        self.benchmark     = benchmark.dropna() if benchmark is not None else None
        self.risk_free_rate = risk_free_rate
        self.trading_days  = trading_days
        self.daily_rf      = (1 + risk_free_rate) ** (1 / trading_days) - 1

    # ------------------------------------------------------------------
    # Core return metrics
    # ------------------------------------------------------------------

    def total_return(self) -> float:
        return float((1 + self.returns).prod() - 1)

    def annualized_return(self) -> float:
        n = len(self.returns)
        return float((1 + self.total_return()) ** (self.trading_days / n) - 1)

    def annualized_volatility(self) -> float:
        return float(self.returns.std() * np.sqrt(self.trading_days))

    def cagr(self) -> float:
        """Compound Annual Growth Rate."""
        return self.annualized_return()

    # ------------------------------------------------------------------
    # Risk-adjusted ratios
    # ------------------------------------------------------------------

    def sharpe_ratio(self) -> float:
        """
        Annualised Sharpe Ratio.
        SR = (E[R] - Rf) / σ(R) * √252
        """
        excess = self.returns - self.daily_rf
        return float(excess.mean() / (excess.std() + 1e-10) * np.sqrt(self.trading_days))

    def sortino_ratio(self, mar: float = 0.0) -> float:
        """
        Sortino Ratio — penalises only downside volatility.
        SR_sortino = (E[R] - MAR) / σ_downside
        """
        daily_mar = (1 + mar) ** (1 / self.trading_days) - 1
        excess    = self.returns - daily_mar
        downside  = excess[excess < 0].std() * np.sqrt(self.trading_days)
        ann_ret   = self.annualized_return() - mar
        return float(ann_ret / (downside + 1e-10))

    def calmar_ratio(self) -> float:
        """Calmar = Annualized Return / Maximum Drawdown."""
        mdd = self.max_drawdown()
        return float(self.annualized_return() / (mdd + 1e-10))

    def omega_ratio(self, threshold: float = 0.0) -> float:
        """
        Omega Ratio = Σ gains above threshold / Σ losses below threshold.
        Ω > 1 means more probability mass above threshold than below.
        """
        daily_thr = (1 + threshold) ** (1 / self.trading_days) - 1
        gains  = (self.returns - daily_thr).clip(lower=0).sum()
        losses = (daily_thr - self.returns).clip(lower=0).sum()
        return float(gains / (losses + 1e-10))

    def tail_ratio(self, percentile: float = 5.0) -> float:
        """
        Tail Ratio = |P95| / |P5|
        Measures asymmetry of tails. > 1.0 = right-skewed (good).
        """
        p95 = abs(np.percentile(self.returns, 100 - percentile))
        p05 = abs(np.percentile(self.returns, percentile))
        return float(p95 / (p05 + 1e-10))

    # ------------------------------------------------------------------
    # Drawdown
    # ------------------------------------------------------------------

    def drawdown_series(self) -> pd.Series:
        wealth = (1 + self.returns).cumprod()
        peak   = wealth.cummax()
        return (wealth - peak) / peak

    def max_drawdown(self) -> float:
        return float(-self.drawdown_series().min())

    def max_drawdown_duration(self) -> int:
        """Longest consecutive days spent in drawdown (underwater)."""
        dd = self.drawdown_series()
        in_dd = (dd < 0).astype(int)
        max_dur = 0
        cur_dur = 0
        for x in in_dd:
            if x:
                cur_dur += 1
                max_dur = max(max_dur, cur_dur)
            else:
                cur_dur = 0
        return max_dur

    def recovery_factor(self) -> float:
        """Total Return / Max Drawdown."""
        return float(abs(self.total_return()) / (self.max_drawdown() + 1e-10))

    # ------------------------------------------------------------------
    # Benchmark-relative metrics
    # ------------------------------------------------------------------

    def alpha_beta(self) -> Dict[str, float]:
        """
        Compute Jensen's Alpha and Market Beta via OLS regression.

        Alpha is annualised; Beta is dimensionless.
        """
        if self.benchmark is None:
            return {"alpha": np.nan, "beta": np.nan}

        p, bm = self.returns.align(self.benchmark, join="inner")
        excess_p  = p  - self.daily_rf
        excess_bm = bm - self.daily_rf

        beta, alpha, r, p_val, se = stats.linregress(
            excess_bm.values, excess_p.values
        )
        alpha_ann = alpha * self.trading_days
        return {
            "alpha":   round(float(alpha_ann), 4),
            "beta":    round(float(beta), 4),
            "r_squared": round(float(r ** 2), 4),
        }

    def information_ratio(self) -> float:
        """
        IR = mean(active return) / tracking error.
        Measures consistency of outperformance vs. benchmark.
        """
        if self.benchmark is None:
            return np.nan
        p, bm = self.returns.align(self.benchmark, join="inner")
        active = p - bm
        return float(active.mean() / (active.std() + 1e-10) * np.sqrt(self.trading_days))

    def tracking_error(self) -> float:
        """Annualised standard deviation of active returns."""
        if self.benchmark is None:
            return np.nan
        p, bm = self.returns.align(self.benchmark, join="inner")
        return float((p - bm).std() * np.sqrt(self.trading_days))

    def up_down_capture(self) -> Dict[str, float]:
        """
        Up-capture: return in up markets vs benchmark.
        Down-capture: return in down markets vs benchmark.
        Ideal: high up-capture, low down-capture.
        """
        if self.benchmark is None:
            return {"up_capture": np.nan, "down_capture": np.nan}

        p, bm = self.returns.align(self.benchmark, join="inner")
        up_days   = bm > 0
        down_days = bm < 0

        up_cap   = p[up_days].mean()   / (bm[up_days].mean()   + 1e-10)
        down_cap = p[down_days].mean() / (bm[down_days].mean() + 1e-10)

        return {
            "up_capture":   round(float(up_cap), 4),
            "down_capture": round(float(down_cap), 4),
        }

    # ------------------------------------------------------------------
    # Rolling metrics
    # ------------------------------------------------------------------

    def rolling_sharpe(self, window: int = 252) -> pd.Series:
        excess = self.returns - self.daily_rf
        return (
            excess.rolling(window).mean()
            / (excess.rolling(window).std() + 1e-10)
            * np.sqrt(self.trading_days)
        ).rename("rolling_sharpe")

    def rolling_volatility(self, window: int = 63) -> pd.Series:
        return (
            self.returns.rolling(window).std() * np.sqrt(self.trading_days)
        ).rename("rolling_vol")

    def rolling_max_drawdown(self, window: int = 252) -> pd.Series:
        def _mdd(r):
            wealth = (1 + pd.Series(r)).cumprod()
            return -(wealth / wealth.cummax() - 1).min()
        return self.returns.rolling(window).apply(_mdd, raw=False).rename("rolling_mdd")

    # ------------------------------------------------------------------
    # Full tearsheet
    # ------------------------------------------------------------------

    def tearsheet(self) -> pd.DataFrame:
        """
        Return a comprehensive performance tearsheet as a DataFrame.
        """
        ab = self.alpha_beta()
        uc = self.up_down_capture()

        metrics = {
            "Total Return":          f"{self.total_return():.2%}",
            "CAGR":                  f"{self.cagr():.2%}",
            "Annualized Volatility": f"{self.annualized_volatility():.2%}",
            "Sharpe Ratio":          f"{self.sharpe_ratio():.2f}",
            "Sortino Ratio":         f"{self.sortino_ratio():.2f}",
            "Calmar Ratio":          f"{self.calmar_ratio():.2f}",
            "Omega Ratio":           f"{self.omega_ratio():.2f}",
            "Tail Ratio":            f"{self.tail_ratio():.2f}",
            "Max Drawdown":          f"{self.max_drawdown():.2%}",
            "Max DD Duration (days)": self.max_drawdown_duration(),
            "Recovery Factor":       f"{self.recovery_factor():.2f}",
            "Skewness":              f"{stats.skew(self.returns):.2f}",
            "Excess Kurtosis":       f"{stats.kurtosis(self.returns):.2f}",
            "Alpha (ann.)":          f"{ab['alpha']:.2%}" if not np.isnan(ab["alpha"]) else "N/A",
            "Beta":                  f"{ab['beta']:.2f}"  if not np.isnan(ab["beta"])  else "N/A",
            "R²":                    f"{ab['r_squared']:.2%}" if not np.isnan(ab.get("r_squared", np.nan)) else "N/A",
            "Information Ratio":     f"{self.information_ratio():.2f}" if self.benchmark is not None else "N/A",
            "Tracking Error":        f"{self.tracking_error():.2%}"    if self.benchmark is not None else "N/A",
            "Up Capture":            f"{uc['up_capture']:.2%}"   if not np.isnan(uc["up_capture"])   else "N/A",
            "Down Capture":          f"{uc['down_capture']:.2%}" if not np.isnan(uc["down_capture"]) else "N/A",
        }

        return pd.DataFrame.from_dict(metrics, orient="index", columns=["Value"])
