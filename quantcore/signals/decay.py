"""
Factor decay and signal horizon analysis.

Measures how predictive power (IC) of a signal degrades
over different forward-return horizons — original implementation
inspired by academic factor research (Grinold & Kahn, Active Portfolio Management).
"""

import numpy as np
import pandas as pd
from typing import List


class FactorDecayAnalyzer:
    """
    Analyse how fast a factor signal loses its predictive power.

    Parameters
    ----------
    max_horizon : int
        Maximum forward-return horizon in days to test (default 63).
    horizons : list, optional
        Specific horizons to test. Overrides max_horizon if provided.
    """

    def __init__(
        self,
        max_horizon: int = 63,
        horizons: List[int] = None,
    ):
        self.horizons = horizons or list(range(1, max_horizon + 1, 2))

    def ic_decay(
        self,
        factor_scores: pd.DataFrame,
        prices: pd.DataFrame,
        method: str = "spearman",
    ) -> pd.DataFrame:
        """
        Compute Information Coefficient at each forward horizon.

        For each horizon h, compute rank IC between today's factor score
        and the h-day forward return. Shows how quickly the signal decays.

        Parameters
        ----------
        factor_scores : pd.DataFrame
            Factor scores (rows=dates, cols=assets).
        prices : pd.DataFrame
            Price DataFrame aligned with factor_scores.
        method : 'spearman' or 'pearson'

        Returns
        -------
        pd.DataFrame with columns [horizon, mean_ic, std_ic, ir, t_stat]
        """
        returns = np.log(prices / prices.shift(1))
        records = []

        for h in self.horizons:
            fwd_ret = returns.rolling(h).sum().shift(-h)
            ic_vals = []

            for date in factor_scores.index:
                if date not in fwd_ret.index:
                    continue
                f = factor_scores.loc[date].dropna()
                r = fwd_ret.loc[date].reindex(f.index).dropna()
                f, r = f.align(r, join="inner")
                if len(f) < 5:
                    continue
                ic = f.corr(r, method=method)
                if not np.isnan(ic):
                    ic_vals.append(ic)

            if not ic_vals:
                continue

            ic_arr  = np.array(ic_vals)
            mean_ic = ic_arr.mean()
            std_ic  = ic_arr.std()
            ir      = mean_ic / (std_ic + 1e-8)
            t_stat  = mean_ic / (std_ic / np.sqrt(len(ic_arr)) + 1e-8)

            records.append({
                "horizon": h,
                "mean_ic": round(mean_ic, 4),
                "std_ic":  round(std_ic,  4),
                "ir":      round(ir,      4),
                "t_stat":  round(t_stat,  4),
                "n_obs":   len(ic_arr),
            })

        return pd.DataFrame(records).set_index("horizon")

    def rolling_ic_surface(
        self,
        factor_scores: pd.DataFrame,
        prices: pd.DataFrame,
        horizons: List[int] = None,
        window: int = 63,
    ) -> pd.DataFrame:
        """
        Compute a 2D surface of rolling IC over time AND horizon.

        Returns a DataFrame where:
          - rows = dates (sampled monthly)
          - columns = horizons
          - values = rolling IC at that date and horizon

        Used to generate the 3D IC surface plot.
        """
        horizons   = horizons or self.horizons[:10]
        returns    = np.log(prices / prices.shift(1))
        date_index = factor_scores.resample("ME").last().index

        surface = pd.DataFrame(index=date_index, columns=horizons, dtype=float)

        for h in horizons:
            fwd_ret = returns.rolling(h).sum().shift(-h)
            daily_ic = []
            daily_dates = []

            dates = factor_scores.index
            for i, date in enumerate(dates):
                if date not in fwd_ret.index:
                    continue
                start = max(0, i - window)
                window_scores = factor_scores.iloc[start:i]
                window_fwd    = fwd_ret.reindex(window_scores.index)

                ic_vals = []
                for d in window_scores.index:
                    f = window_scores.loc[d].dropna()
                    r = window_fwd.loc[d].reindex(f.index).dropna()
                    f, r = f.align(r, join="inner")
                    if len(f) < 5:
                        continue
                    ic = f.corr(r, method="spearman")
                    if not np.isnan(ic):
                        ic_vals.append(ic)

                if ic_vals:
                    daily_ic.append(np.mean(ic_vals))
                    daily_dates.append(date)

            ic_series = pd.Series(daily_ic, index=daily_dates)
            for d in date_index:
                if d in ic_series.index:
                    surface.loc[d, h] = ic_series[d]

        return surface.dropna(how="all").astype(float)

    def half_life(self, ic_decay_df: pd.DataFrame) -> float:
        """
        Estimate signal half-life: horizon at which IC drops to half its peak.

        Returns number of days.
        """
        if ic_decay_df.empty:
            return np.nan
        ic = ic_decay_df["mean_ic"].abs()
        peak = ic.max()
        half = peak / 2
        below = ic[ic <= half]
        if below.empty:
            return float(ic.index[-1])
        return float(below.index[0])

    def turnover_decay(
        self,
        factor_scores: pd.DataFrame,
        lag: int = 1,
    ) -> pd.Series:
        """
        Measure factor autocorrelation (rank correlation with lagged self).

        High autocorrelation = slow-moving signal = lower turnover.
        Returns a Series of autocorrelations at lags 1..21.
        """
        lags = range(1, 22)
        autocorrs = {}
        for l in lags:
            vals = []
            for col in factor_scores.columns:
                s = factor_scores[col].dropna()
                if len(s) > l + 10:
                    vals.append(s.corr(s.shift(l), method="spearman"))
            autocorrs[l] = np.nanmean(vals) if vals else np.nan
        return pd.Series(autocorrs, name="autocorrelation")
