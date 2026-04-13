"""
Position sizing.

Implements:
  - Kelly Criterion (full, fractional, and multi-asset)
  - Volatility targeting
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from typing import Optional


class KellyCriterion:
    """
    Kelly Criterion for optimal position sizing.

    The Kelly fraction maximises the expected log growth rate of capital.
    In practice, fractional Kelly (e.g. 0.25–0.5) is used to reduce
    variance and drawdowns.

    Parameters
    ----------
    fraction : float
        Fraction of full Kelly to use (default 0.5 = half-Kelly).
    """

    def __init__(self, fraction: float = 0.5):
        self.fraction = fraction

    # ------------------------------------------------------------------
    # Single bet Kelly
    # ------------------------------------------------------------------

    def single_bet(
        self,
        win_prob: float,
        win_return: float,
        loss_return: float = -1.0,
    ) -> float:
        """
        Classic Kelly formula for a single binary bet.

        f* = p/a - q/b
        where p = win prob, q = 1 - p, a = loss fraction, b = win fraction.

        Returns
        -------
        float : fraction of capital to bet (fractional Kelly applied).
        """
        if win_prob <= 0 or win_prob >= 1:
            raise ValueError("win_prob must be in (0, 1)")
        q = 1 - win_prob
        b = abs(win_return)
        a = abs(loss_return)
        kelly = (win_prob / a) - (q / b)
        return float(np.clip(kelly * self.fraction, 0, 1))

    # ------------------------------------------------------------------
    # Continuous Kelly from return series
    # ------------------------------------------------------------------

    def continuous_kelly(
        self,
        returns: pd.Series,
        lookback: Optional[int] = None,
    ) -> float:
        """
        Kelly fraction from a continuous return distribution.

        For a normal distribution: f* = mu / sigma²

        Uses rolling window if lookback is provided.

        Returns
        -------
        float : fractional Kelly position size as fraction of capital.
        """
        r = returns.dropna()
        if lookback:
            r = r.iloc[-lookback:]
        mu    = float(r.mean())
        sigma2 = float(r.var())
        if sigma2 < 1e-12:
            return 0.0
        kelly = mu / sigma2
        return float(np.clip(kelly * self.fraction, -1, 1))

    # ------------------------------------------------------------------
    # Multi-asset Kelly
    # ------------------------------------------------------------------

    def multi_asset(
        self,
        returns: pd.DataFrame,
        allow_short: bool = False,
    ) -> pd.Series:
        """
        Multi-asset Kelly — maximise expected log return.

        Solved as:  max  mu' w - 0.5 * w' Sigma w
        which gives: w* = Sigma^{-1} * mu

        Normalised to sum to 1 after fractional scaling.

        Returns
        -------
        pd.Series of Kelly weights (fractional).
        """
        mu    = returns.mean().values
        sigma = returns.cov().values

        try:
            kelly_full = np.linalg.solve(sigma, mu)
        except np.linalg.LinAlgError:
            kelly_full = np.linalg.lstsq(sigma, mu, rcond=None)[0]

        kelly = kelly_full * self.fraction

        if not allow_short:
            kelly = np.clip(kelly, 0, None)

        total = np.abs(kelly).sum()
        if total > 1e-8:
            kelly = kelly / total

        return pd.Series(kelly, index=returns.columns)

    # ------------------------------------------------------------------
    # Volatility targeting
    # ------------------------------------------------------------------

    @staticmethod
    def volatility_target(
        returns: pd.Series,
        target_vol: float = 0.10,
        lookback: int = 63,
        trading_days: int = 252,
    ) -> pd.Series:
        """
        Scale position size to target a given annualised volatility.

        leverage_t = target_vol / realized_vol_t

        Returns a time series of leverage multipliers (capped at 2.0).
        """
        realized_vol = (
            returns.rolling(lookback).std() * np.sqrt(trading_days)
        )
        leverage = target_vol / (realized_vol + 1e-8)
        return leverage.clip(0, 2.0)

    # ------------------------------------------------------------------
    # Rolling Kelly
    # ------------------------------------------------------------------

    def rolling_kelly(
        self,
        returns: pd.Series,
        window: int = 252,
    ) -> pd.Series:
        """
        Compute rolling fractional Kelly position size.

        Returns a Series of position sizes over time.
        """
        return returns.rolling(window).apply(
            lambda r: self.continuous_kelly(pd.Series(r)),
            raw=False,
        )
