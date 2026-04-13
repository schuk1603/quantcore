"""
Mean-reversion signal generation.

Implements:
  - Z-score mean reversion (rolling window)
  - Bollinger Bands signal
  - Ornstein-Uhlenbeck half-life estimation
  - RSI (Relative Strength Index)
"""

import numpy as np
import pandas as pd
from scipy.stats import linregress


class MeanReversionSignal:
    """
    Compute mean-reversion alpha signals from a price or return series.

    Parameters
    ----------
    window : int
        Rolling window in trading days for z-score computation (default 63).
    """

    def __init__(self, window: int = 63):
        self.window = window

    # ------------------------------------------------------------------
    # Z-score signal
    # ------------------------------------------------------------------

    def zscore(self, prices: pd.DataFrame) -> pd.DataFrame:
        """
        Rolling z-score of log prices.

        signal = (log_price - rolling_mean) / rolling_std

        Negative signal → price is above mean → short.
        Positive signal → price is below mean → long.
        Returns negated z-score so positive = expected upside.
        """
        log_p = np.log(prices)
        mu    = log_p.rolling(self.window).mean()
        sigma = log_p.rolling(self.window).std()
        z     = (log_p - mu) / (sigma + 1e-8)
        return (-z).dropna(how="all")          # negate: below mean → long

    # ------------------------------------------------------------------
    # Bollinger Bands
    # ------------------------------------------------------------------

    def bollinger_bands(
        self,
        prices: pd.DataFrame,
        n_std: float = 2.0,
    ) -> pd.DataFrame:
        """
        Bollinger Bands signal.

        Returns continuous signal in [-1, 1]:
          +1 when price touches lower band (oversold → long)
          -1 when price touches upper band (overbought → short)
          0  at the middle band
        """
        mu    = prices.rolling(self.window).mean()
        sigma = prices.rolling(self.window).std()
        upper = mu + n_std * sigma
        lower = mu - n_std * sigma

        width = upper - lower + 1e-8
        # Linear interpolation between bands; clipped to [-1, 1]
        signal = 2 * (mu - prices) / width
        return signal.clip(-1, 1).dropna(how="all")

    # ------------------------------------------------------------------
    # RSI
    # ------------------------------------------------------------------

    def rsi(self, prices: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """
        Relative Strength Index (Wilder).

        Returns mean-reversion signal:
          +1 for RSI < 30 (oversold → expect rebound)
          -1 for RSI > 70 (overbought → expect decline)
           0 for neutral zone
        """
        delta  = prices.diff()
        gain   = delta.clip(lower=0).rolling(period).mean()
        loss   = (-delta.clip(upper=0)).rolling(period).mean()
        rs     = gain / (loss + 1e-8)
        rsi    = 100 - 100 / (1 + rs)

        # Convert to [-1, 1] signal: oversold positive, overbought negative
        signal = -(rsi - 50) / 50
        return signal.dropna(how="all")

    # ------------------------------------------------------------------
    # Ornstein-Uhlenbeck half-life
    # ------------------------------------------------------------------

    @staticmethod
    def ou_half_life(series: pd.Series) -> float:
        """
        Estimate the half-life of mean reversion for a series using
        the Ornstein-Uhlenbeck regression:

            Δy_t = α + β * y_{t-1} + ε_t

        Half-life = -ln(2) / β   (in the same units as the series index)

        A shorter half-life → faster mean reversion → better signal.
        Returns half-life in bars (days if daily data).
        """
        y     = series.dropna()
        delta = y.diff().dropna()
        y_lag = y.shift(1).dropna()
        y_lag, delta = y_lag.align(delta, join="inner")

        slope, intercept, r, p, se = linregress(y_lag.values, delta.values)

        if slope >= 0:
            return np.inf       # non-mean-reverting

        return -np.log(2) / slope

    def mean_reverting_score(self, prices: pd.DataFrame) -> pd.DataFrame:
        """
        For each asset, compute OU half-life over a rolling window.

        Returns a DataFrame of half-lives (lower = stronger mean reversion).
        """
        log_p = np.log(prices)
        half_lives = {}
        for col in log_p.columns:
            hl = log_p[col].rolling(self.window * 2).apply(
                lambda x: self.ou_half_life(pd.Series(x)), raw=False
            )
            half_lives[col] = hl
        return pd.DataFrame(half_lives, index=log_p.index)

    # ------------------------------------------------------------------
    # Aggregate signal
    # ------------------------------------------------------------------

    def combined(self, prices: pd.DataFrame) -> pd.DataFrame:
        """
        Equal-weight of z-score and Bollinger Bands signals, normalized.
        """
        z  = self.zscore(prices)
        bb = self.bollinger_bands(prices)
        rs = self.rsi(prices).reindex(z.index)

        signal = (z + bb.reindex(z.index) + rs) / 3.0
        mu  = signal.mean(axis=1)
        std = signal.std(axis=1)
        return signal.sub(mu, axis=0).div(std + 1e-8, axis=0).dropna(how="all")
