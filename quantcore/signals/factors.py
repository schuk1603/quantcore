"""
Multi-factor alpha model.

Implements a cross-sectional factor model with:
  - Value      (book-to-market proxy via trailing P/E reversal)
  - Momentum   (12-1 month return)
  - Quality    (low volatility / high Sharpe)
  - Size       (negative log-market cap proxy via price * volume)
  - Low Vol    (low trailing realized volatility)

Each factor is cross-sectionally z-scored and combined into a composite
alpha score used for portfolio construction.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from typing import Dict, Optional


class FactorModel:
    """
    Cross-sectional factor model for equity alpha generation.

    Parameters
    ----------
    lookback_mom : int
        Lookback for momentum factor (default 252 days).
    lookback_vol : int
        Lookback for volatility/quality factor (default 63 days).
    factor_weights : dict, optional
        Custom weights for each factor. Defaults to equal weighting.
    """

    FACTORS = ["momentum", "low_vol", "quality", "reversal"]

    def __init__(
        self,
        lookback_mom: int = 252,
        lookback_vol: int = 63,
        factor_weights: Optional[Dict[str, float]] = None,
    ):
        self.lookback_mom = lookback_mom
        self.lookback_vol = lookback_vol
        self.factor_weights = factor_weights or {f: 1.0 for f in self.FACTORS}

    # ------------------------------------------------------------------
    # Individual factor scores
    # ------------------------------------------------------------------

    def momentum_factor(self, prices: pd.DataFrame, skip: int = 21) -> pd.DataFrame:
        """12-1 month cross-sectional return rank."""
        ret = prices.shift(skip) / prices.shift(self.lookback_mom) - 1
        return self._cross_section_zscore(ret)

    def low_vol_factor(self, returns: pd.DataFrame) -> pd.DataFrame:
        """Negative rolling realized volatility (low vol = positive score)."""
        vol = returns.rolling(self.lookback_vol).std() * np.sqrt(252)
        return self._cross_section_zscore(-vol)

    def quality_factor(self, returns: pd.DataFrame) -> pd.DataFrame:
        """Rolling Sharpe ratio as a quality proxy."""
        mu    = returns.rolling(self.lookback_vol).mean() * 252
        sigma = returns.rolling(self.lookback_vol).std() * np.sqrt(252)
        sharpe = mu / (sigma + 1e-8)
        return self._cross_section_zscore(sharpe)

    def reversal_factor(self, returns: pd.DataFrame, window: int = 5) -> pd.DataFrame:
        """Short-term reversal: negative trailing 1-week return."""
        short_ret = returns.rolling(window).sum()
        return self._cross_section_zscore(-short_ret)

    # ------------------------------------------------------------------
    # Composite alpha
    # ------------------------------------------------------------------

    def composite_alpha(
        self,
        prices: pd.DataFrame,
        returns: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Weighted combination of all factor scores.

        Returns a cross-sectionally z-scored composite alpha DataFrame.
        Each row sums to ~0 (long-short neutral within the universe).
        """
        scores = {
            "momentum":  self.momentum_factor(prices),
            "low_vol":   self.low_vol_factor(returns),
            "quality":   self.quality_factor(returns),
            "reversal":  self.reversal_factor(returns),
        }

        total_w = sum(self.factor_weights.values())
        composite = None

        for name, score in scores.items():
            w = self.factor_weights.get(name, 1.0) / total_w
            if composite is None:
                composite = score * w
            else:
                composite = composite.add(score * w, fill_value=0)

        return self._cross_section_zscore(composite).dropna(how="all")

    # ------------------------------------------------------------------
    # Factor IC / IR (Information Coefficient)
    # ------------------------------------------------------------------

    def information_coefficient(
        self,
        factor_scores: pd.DataFrame,
        forward_returns: pd.DataFrame,
        periods: int = 21,
    ) -> pd.Series:
        """
        Compute rolling rank IC (Spearman correlation) between factor
        scores and forward returns.

        IC > 0.05 is considered economically meaningful.
        IC > 0.10 is considered strong.

        Returns
        -------
        pd.Series of daily IC values.
        """
        fwd_ret = forward_returns.shift(-periods)
        ic_series = []
        dates = factor_scores.index.intersection(fwd_ret.index)

        for date in dates:
            f = factor_scores.loc[date].dropna()
            r = fwd_ret.loc[date].reindex(f.index).dropna()
            if len(r) < 5:
                ic_series.append(np.nan)
                continue
            f, r = f.align(r, join="inner")
            ic = f.corr(r, method="spearman")
            ic_series.append(ic)

        return pd.Series(ic_series, index=dates, name="IC")

    def information_ratio(self, ic_series: pd.Series) -> float:
        """IR = mean(IC) / std(IC). Target > 0.5 for viable alpha."""
        clean = ic_series.dropna()
        return float(clean.mean() / (clean.std() + 1e-8))

    # ------------------------------------------------------------------
    # Factor attribution (Brinson-Hood-Beebower style)
    # ------------------------------------------------------------------

    def factor_attribution(
        self,
        portfolio_returns: pd.Series,
        factor_returns: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Regress portfolio returns on factor returns to decompose attribution.

        Returns a DataFrame with factor exposures (betas), t-stats,
        and explained variance (R²).
        """
        clean = pd.concat([portfolio_returns, factor_returns], axis=1, join="inner").dropna()
        y = clean.iloc[:, 0].values
        X = clean.iloc[:, 1:].values

        # Add intercept
        X_with_const = np.column_stack([np.ones(len(X)), X])
        betas, resid, rank, sv = np.linalg.lstsq(X_with_const, y, rcond=None)

        y_hat = X_with_const @ betas
        ss_res = np.sum((y - y_hat) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        r2 = 1 - ss_res / (ss_tot + 1e-12)

        # Standard errors
        n, k = X_with_const.shape
        mse = ss_res / max(n - k, 1)
        cov_betas = mse * np.linalg.pinv(X_with_const.T @ X_with_const)
        se = np.sqrt(np.diag(cov_betas))
        t_stats = betas / (se + 1e-12)

        factor_names = ["alpha"] + list(factor_returns.columns)
        return pd.DataFrame({
            "factor":  factor_names,
            "beta":    betas,
            "t_stat":  t_stats,
            "r_squared": [r2] + [np.nan] * (len(betas) - 1),
        }).set_index("factor")

    # ------------------------------------------------------------------
    # Helper
    # ------------------------------------------------------------------

    @staticmethod
    def _cross_section_zscore(df: pd.DataFrame) -> pd.DataFrame:
        mu  = df.mean(axis=1)
        std = df.std(axis=1)
        return df.sub(mu, axis=0).div(std + 1e-8, axis=0)
