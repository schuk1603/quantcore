"""
Trend-following signals.

Based on publicly documented methods used by CTA/systematic funds
(Man AHL, Winton, Campbell & Company).

Key references:
  - Moskowitz, Ooi, Pedersen (2012) "Time Series Momentum" - Journal of Financial Economics
  - Hurst, Ooi, Pedersen (2017) "A Century of Evidence on Trend-Following Investing" - AQR
  - Baz et al. (2015) "Dissecting Investment Strategies in the Cross Section and Time Series" - PIMCO

All formulas are derived from public academic literature. Original implementation.
"""

import numpy as np
import pandas as pd
from typing import List, Optional


class TrendFollowingSignal:
    """
    EWMA-based trend-following signal — the foundation of systematic CTA funds.

    The core insight (Moskowitz et al. 2012): assets that have gone up over
    the past 12 months tend to continue going up over the next month.
    This holds across asset classes and centuries of data.

    Strategy: go long assets with positive trailing return,
    short assets with negative trailing return,
    scaled by realised volatility (so each position contributes
    equal risk regardless of how volatile the asset is).
    """

    def __init__(
        self,
        fast_span:  int = 8,
        slow_span:  int = 24,
        trend_span: int = 252,
        vol_target: float = 0.15,
        trading_days: int = 252,
    ):
        self.fast_span   = fast_span
        self.slow_span   = slow_span
        self.trend_span  = trend_span
        self.vol_target  = vol_target
        self.trading_days = trading_days

    # ------------------------------------------------------------------
    # EWMA crossover (Man AHL / Campbell style)
    # ------------------------------------------------------------------

    def ewma_crossover(self, prices: pd.DataFrame) -> pd.DataFrame:
        """
        Exponentially weighted moving average crossover signal.

        signal = (EWMA_fast - EWMA_slow) / price

        Positive = uptrend (long), Negative = downtrend (short).
        Volatility-normalised so signals are comparable across assets.

        This is the core building block of most CTA trend systems.
        """
        fast = prices.ewm(span=self.fast_span,  adjust=False).mean()
        slow = prices.ewm(span=self.slow_span,  adjust=False).mean()

        raw_signal = (fast - slow) / (prices + 1e-8)

        # Volatility-scale each signal
        returns    = np.log(prices / prices.shift(1))
        vol        = returns.ewm(span=63).std() * np.sqrt(self.trading_days)
        scaled     = raw_signal / (vol + 1e-8)

        return scaled.dropna(how="all")

    # ------------------------------------------------------------------
    # Time-series momentum (Moskowitz, Ooi, Pedersen 2012)
    # ------------------------------------------------------------------

    def tsmom(
        self,
        returns: pd.DataFrame,
        lookback: int = 252,
        skip: int = 21,
        vol_window: int = 63,
    ) -> pd.DataFrame:
        """
        Time-series momentum signal.

        For each asset independently:
          signal_t = sign(r_{t-skip, t-lookback}) * (vol_target / sigma_t)

        Where sigma_t is the trailing realised volatility.
        This gives a +1/-1 direction scaled by vol target.

        The key insight: past 12-month return (skipping last month)
        predicts next-month return with t-stat > 2 across all asset classes.
        """
        cum_ret = returns.rolling(lookback - skip).sum().shift(skip)
        direction = np.sign(cum_ret)

        vol = returns.rolling(vol_window).std() * np.sqrt(self.trading_days)
        position_size = self.vol_target / (vol + 1e-8)
        position_size = position_size.clip(upper=2.0)   # max 2x leverage

        signal = direction * position_size
        return signal.dropna(how="all")

    # ------------------------------------------------------------------
    # Breakout signal (Donchian channel)
    # ------------------------------------------------------------------

    def breakout(self, prices: pd.DataFrame, window: int = 252) -> pd.DataFrame:
        """
        Donchian channel breakout — price at N-day high/low.

        signal = (price - midpoint) / (high - low + 1e-8)
        Ranges from -0.5 (at N-day low) to +0.5 (at N-day high).

        Used by systematic trend funds as a complementary signal to EWMA.
        """
        high     = prices.rolling(window).max()
        low      = prices.rolling(window).min()
        midpoint = (high + low) / 2
        width    = high - low + 1e-8
        signal   = (prices - midpoint) / width
        return signal.dropna(how="all")

    # ------------------------------------------------------------------
    # Multi-speed trend combination (AHL diversified style)
    # ------------------------------------------------------------------

    def combined(
        self,
        prices: pd.DataFrame,
        returns: pd.DataFrame,
        speeds: List[int] = None,
    ) -> pd.DataFrame:
        """
        Combine trend signals across multiple speeds.

        Funds like Man AHL run the same trend strategy across ~20 different
        lookback windows and blend them. This improves robustness — no single
        lookback dominates, and different speeds catch different trend lengths.

        Speeds are (fast_span, slow_span) pairs.
        """
        speeds = speeds or [(4, 16), (8, 32), (16, 64), (32, 128)]
        signals = []

        for fast, slow in speeds:
            orig_fast, orig_slow = self.fast_span, self.slow_span
            self.fast_span = fast
            self.slow_span = slow
            sig = self.ewma_crossover(prices)
            signals.append(sig)
            self.fast_span = orig_fast
            self.slow_span = orig_slow

        combined = pd.concat(signals).groupby(level=0).mean()

        # Add TSMOM signal
        tsmom = self.tsmom(returns)
        combined = combined.add(tsmom.reindex(combined.index), fill_value=0) / 2

        # Cross-sectional normalise
        mu  = combined.mean(axis=1)
        std = combined.std(axis=1)
        return combined.sub(mu, axis=0).div(std + 1e-8, axis=0).dropna(how="all")
