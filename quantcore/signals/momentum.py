"""
Momentum signal generation.

Implements:
  - Cross-sectional momentum  (rank assets by trailing return)
  - Time-series momentum      (trend-following, Moskowitz et al. 2012)
  - Dual momentum             (absolute + relative, Antonacci 2014)
  - Risk-adjusted momentum    (Sharpe-weighted signal)
"""

import numpy as np
import pandas as pd


class MomentumSignal:
    """
    Compute momentum-based alpha signals from a price DataFrame.

    Parameters
    ----------
    lookback : int
        Lookback window in trading days (default 252 = 12 months).
    skip : int
        Days to skip before the lookback window to avoid short-term reversal
        (default 21 = 1 month).
    """

    def __init__(self, lookback: int = 252, skip: int = 21):
        self.lookback = lookback
        self.skip = skip

    # ------------------------------------------------------------------
    # Cross-sectional momentum
    # ------------------------------------------------------------------

    def cross_sectional(self, prices: pd.DataFrame) -> pd.DataFrame:
        """
        Rank assets each day by their (lookback - skip) return.

        Returns z-scored ranks in [-1, 1] range.
        """
        ret = prices.shift(self.skip) / prices.shift(self.lookback) - 1
        ranks = ret.rank(axis=1, pct=True)        # [0,1] percentile rank
        signal = 2 * ranks - 1                    # map to [-1, 1]
        return signal.dropna(how="all")

    # ------------------------------------------------------------------
    # Time-series momentum
    # ------------------------------------------------------------------

    def time_series(self, prices: pd.DataFrame) -> pd.DataFrame:
        """
        Sign of trailing (lookback - skip) return per asset.

        Returns +1 (long) / -1 (short) signal.
        """
        ret = prices.shift(self.skip) / prices.shift(self.lookback) - 1
        return np.sign(ret).dropna(how="all")

    # ------------------------------------------------------------------
    # Dual momentum (Antonacci)
    # ------------------------------------------------------------------

    def dual_momentum(
        self,
        prices: pd.DataFrame,
        risk_free_rate: float = 0.0,
        trading_days: int = 252,
    ) -> pd.DataFrame:
        """
        Dual momentum: only go long if:
          1. Absolute: trailing return > risk-free rate  (time-series filter)
          2. Relative: asset is top cross-sectional performer

        Returns signal in {-1, 0, 1}.
        """
        daily_rf = (1 + risk_free_rate) ** (1 / trading_days) - 1
        trailing_rf = (1 + daily_rf) ** (self.lookback - self.skip) - 1

        ret = prices.shift(self.skip) / prices.shift(self.lookback) - 1

        absolute = (ret > trailing_rf).astype(float)
        relative = (ret.rank(axis=1, pct=True) >= 0.5).astype(float)

        signal = absolute * relative
        signal[signal == 0] = -1        # short the rest
        return signal.dropna(how="all")

    # ------------------------------------------------------------------
    # Risk-adjusted momentum (Sharpe-weighted)
    # ------------------------------------------------------------------

    def sharpe_weighted(
        self,
        returns: pd.DataFrame,
        vol_window: int = 63,
    ) -> pd.DataFrame:
        """
        Scale momentum signals by each asset's rolling Sharpe ratio.

        signal = rolling_mean(returns) / rolling_std(returns)

        Cross-sectionally z-scored at each date.
        """
        roll_mean = returns.rolling(vol_window).mean()
        roll_std  = returns.rolling(vol_window).std()
        sharpe    = roll_mean / (roll_std + 1e-8)

        # Cross-sectional z-score
        mu  = sharpe.mean(axis=1)
        std = sharpe.std(axis=1)
        zscore = sharpe.sub(mu, axis=0).div(std + 1e-8, axis=0)
        return zscore.dropna(how="all")

    # ------------------------------------------------------------------
    # Aggregate signal
    # ------------------------------------------------------------------

    def combined(self, prices: pd.DataFrame, returns: pd.DataFrame) -> pd.DataFrame:
        """
        Equal-weight average of cross-sectional, time-series, and
        Sharpe-weighted signals, z-scored at each date.
        """
        cs = self.cross_sectional(prices)
        ts = self.time_series(prices).reindex(cs.index)
        sw = self.sharpe_weighted(returns).reindex(cs.index)

        combined = (cs + ts + sw) / 3.0
        mu  = combined.mean(axis=1)
        std = combined.std(axis=1)
        return combined.sub(mu, axis=0).div(std + 1e-8, axis=0).dropna(how="all")
