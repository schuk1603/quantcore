"""
Statistical arbitrage — pairs trading.

Implements:
  - Engle-Granger cointegration test for pair selection
  - Kalman Filter for dynamic (time-varying) hedge ratio estimation
  - Spread z-score signal with configurable entry/exit thresholds
"""

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import coint, adfuller
from typing import Tuple, List


class PairsTradingSignal:
    """
    Identify cointegrated pairs and generate trading signals via
    a Kalman Filter that tracks a time-varying hedge ratio.

    Parameters
    ----------
    entry_z : float
        Z-score threshold to open a position (default 2.0).
    exit_z : float
        Z-score threshold to close a position (default 0.5).
    delta : float
        Kalman filter state noise parameter (controls how fast the
        hedge ratio can change). Smaller = slower adaptation.
    """

    def __init__(
        self,
        entry_z: float = 2.0,
        exit_z:  float = 0.5,
        delta:   float = 1e-4,
    ):
        self.entry_z = entry_z
        self.exit_z  = exit_z
        self.delta   = delta

    # ------------------------------------------------------------------
    # Cointegration screening
    # ------------------------------------------------------------------

    def find_cointegrated_pairs(
        self,
        prices: pd.DataFrame,
        pvalue_threshold: float = 0.05,
    ) -> List[Tuple[str, str, float]]:
        """
        Test all pairs for cointegration using the Engle-Granger test.

        Returns
        -------
        List of (asset_A, asset_B, p_value) sorted by p-value ascending.
        """
        tickers = list(prices.columns)
        n = len(tickers)
        results = []

        for i in range(n):
            for j in range(i + 1, n):
                t1 = tickers[i]
                t2 = tickers[j]
                s1 = prices[t1].dropna()
                s2 = prices[t2].dropna()
                combined = pd.concat([s1, s2], axis=1, join="inner")
                if len(combined) < 60:
                    continue
                try:
                    _, pval, _ = coint(combined.iloc[:, 0], combined.iloc[:, 1])
                    if pval < pvalue_threshold:
                        results.append((t1, t2, round(pval, 4)))
                except Exception:
                    continue

        return sorted(results, key=lambda x: x[2])

    # ------------------------------------------------------------------
    # Kalman Filter hedge ratio
    # ------------------------------------------------------------------

    def kalman_hedge_ratio(
        self,
        y: pd.Series,
        x: pd.Series,
    ) -> Tuple[pd.Series, pd.Series]:
        """
        Estimate a time-varying hedge ratio using a 2-state Kalman Filter.

        State vector: theta = [beta, alpha]  (hedge ratio + intercept)
        Observation:  y_t = x_t * beta_t + alpha_t + epsilon_t

        Parameters
        ----------
        y : dependent asset price series
        x : independent asset price series

        Returns
        -------
        hedge_ratio : pd.Series  — time-varying beta
        spread      : pd.Series  — y - beta * x - alpha
        """
        y, x = y.align(x, join="inner")
        n = len(y)

        # Kalman filter matrices
        delta = self.delta
        Vw    = delta / (1 - delta) * np.eye(2)  # state transition noise
        Ve    = 1e-3                              # observation noise variance

        # Initial state
        theta = np.zeros((2, 1))     # [beta, alpha]
        P     = np.zeros((2, 2))     # state covariance

        hedge_ratios = np.zeros(n)
        alphas       = np.zeros(n)
        spreads      = np.zeros(n)

        for t in range(n):
            xt = np.array([[x.iloc[t]], [1.0]])   # observation matrix row

            # Prediction
            P = P + Vw

            # Innovation
            yhat     = float(xt.T @ theta)
            innov    = float(y.iloc[t]) - yhat
            S        = float(xt.T @ P @ xt) + Ve

            # Kalman gain
            K     = P @ xt / S

            # Update
            theta = theta + K * innov
            P     = P - K @ xt.T @ P

            hedge_ratios[t] = theta[0, 0]
            alphas[t]       = theta[1, 0]
            spreads[t]      = float(y.iloc[t]) - hedge_ratios[t] * float(x.iloc[t]) - alphas[t]

        idx = y.index
        return pd.Series(hedge_ratios, index=idx), pd.Series(spreads, index=idx)

    # ------------------------------------------------------------------
    # Signal generation
    # ------------------------------------------------------------------

    def signal(
        self,
        y: pd.Series,
        x: pd.Series,
        spread_window: int = 30,
    ) -> pd.DataFrame:
        """
        Generate trading signals for a cointegrated pair.

        Returns a DataFrame with columns:
          hedge_ratio  : time-varying beta from Kalman Filter
          spread       : y - beta*x - alpha
          zscore       : rolling z-score of spread
          signal_y     : +1 long y / -1 short y / 0 flat
          signal_x     : offsetting position in x
        """
        hedge_ratio, spread = self.kalman_hedge_ratio(y, x)

        # Rolling z-score of spread
        mu    = spread.rolling(spread_window).mean()
        sigma = spread.rolling(spread_window).std()
        z     = (spread - mu) / (sigma + 1e-8)

        # Generate signal
        signal_y = pd.Series(0.0, index=z.index)
        position = 0

        for t in range(len(z)):
            zt = z.iloc[t]
            if np.isnan(zt):
                continue
            if position == 0:
                if zt > self.entry_z:
                    position = -1   # spread too high: short y, long x
                elif zt < -self.entry_z:
                    position = 1    # spread too low:  long y, short x
            elif position == 1 and zt >= -self.exit_z:
                position = 0
            elif position == -1 and zt <= self.exit_z:
                position = 0
            signal_y.iloc[t] = position

        # Hedge: opposite position in x scaled by hedge ratio
        signal_x = -signal_y * hedge_ratio

        return pd.DataFrame({
            "hedge_ratio": hedge_ratio,
            "spread":      spread,
            "zscore":      z,
            "signal_y":    signal_y,
            "signal_x":    signal_x,
        })

    # ------------------------------------------------------------------
    # ADF stationarity check on spread
    # ------------------------------------------------------------------

    @staticmethod
    def adf_test(spread: pd.Series, significance: float = 0.05) -> dict:
        """
        Augmented Dickey-Fuller test on the spread.
        A stationary spread confirms the pair is tradeable.
        """
        result = adfuller(spread.dropna(), autolag="AIC")
        return {
            "adf_stat":   round(result[0], 4),
            "pvalue":     round(result[1], 4),
            "stationary": result[1] < significance,
            "critical_values": result[4],
        }
