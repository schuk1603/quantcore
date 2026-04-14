"""
Carry signal — one of AQR's core published factors.

References:
  - Koijen, Moskowitz, Pedersen, Vrugt (2018)
    "Carry" - Journal of Financial Economics (publicly available)
  - Asness, Moskowitz, Pedersen (2013)
    "Value and Momentum Everywhere" - Journal of Finance
  - AQR white papers on carry (publicly available on aqr.com)

Carry = expected return of an asset assuming prices don't change.
For equities: dividend yield + earnings yield = carry proxy.
For currencies: interest rate differential = carry.
For bonds: yield - duration * yield_change = roll-down carry.

Original implementation based on public academic formulas.
"""

import numpy as np
import pandas as pd
from typing import Optional


class CarrySignal:
    """
    Equity carry signal: dividend yield + earnings yield proxy.

    For equities without direct yield data, we use:
      Carry ≈ Earnings Yield (1/PE) + Dividend Yield + Buyback Yield

    The core idea: assets with higher carry (yield) tend to outperform
    assets with lower carry, after controlling for risk.

    This is one of the most robust factors in academic finance,
    documented across equities, bonds, commodities, and currencies.
    """

    def __init__(self, window: int = 252):
        self.window = window

    # ------------------------------------------------------------------
    # Yield-based carry (equity proxy)
    # ------------------------------------------------------------------

    def earnings_yield_carry(
        self,
        prices: pd.DataFrame,
        earnings_per_share: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Carry = Earnings Yield = EPS / Price = 1 / P/E ratio.

        When EPS data is unavailable (most backtests), we proxy carry
        using the inverse of the trailing P/E estimated from price momentum:
        assets that have de-rated (price fell while earnings held) have
        higher carry (cheaper).

        This is a simplified proxy — real implementations use fundamental data.
        """
        if earnings_per_share is not None:
            carry = earnings_per_share / (prices + 1e-8)
        else:
            # Proxy: 1/PE estimated as mean-reversion of price/rolling-mean ratio
            roll_mean = prices.rolling(self.window).mean()
            pe_proxy  = prices / (roll_mean + 1e-8)
            carry     = 1.0 / (pe_proxy + 1e-8)

        return self._normalise(carry).dropna(how="all")

    # ------------------------------------------------------------------
    # Roll-down carry (futures / term structure proxy)
    # ------------------------------------------------------------------

    def roll_carry(
        self,
        prices: pd.DataFrame,
        short_window: int = 21,
        long_window:  int = 252,
    ) -> pd.DataFrame:
        """
        Roll-down carry: the yield from holding an asset as it rolls
        down the term structure.

        Proxy for equities:
          carry_t = (short-term EMA - long-term EMA) / price

        Positive = price is elevated vs long-run trend = negative carry
        Negative = price is depressed vs long-run trend = positive carry

        Assets with positive carry tend to outperform (Koijen et al. 2018).
        """
        short_ma = prices.ewm(span=short_window).mean()
        long_ma  = prices.ewm(span=long_window).mean()

        # Carry = expected appreciation from mean reversion
        carry = (long_ma - short_ma) / (prices + 1e-8)
        return self._normalise(carry).dropna(how="all")

    # ------------------------------------------------------------------
    # Dividend yield proxy (price stability = dividend proxy)
    # ------------------------------------------------------------------

    def stability_carry(self, returns: pd.DataFrame, window: int = 63) -> pd.DataFrame:
        """
        Stability carry: assets with lower volatility AND positive returns
        are proxying high-yield, stable dividend payers.

        carry_t = rolling_return / rolling_vol
              = rolling Sharpe (annualised)

        This is a quality/carry hybrid signal used by systematic funds.
        """
        roll_ret = returns.rolling(window).mean() * self.window
        roll_vol = returns.rolling(window).std() * np.sqrt(self.window)
        carry    = roll_ret / (roll_vol + 1e-8)
        return self._normalise(carry).dropna(how="all")

    # ------------------------------------------------------------------
    # Combined carry signal
    # ------------------------------------------------------------------

    def combined(
        self,
        prices: pd.DataFrame,
        returns: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Equal-weight blend of carry signals, cross-sectionally normalised.
        """
        c1 = self.earnings_yield_carry(prices)
        c2 = self.roll_carry(prices)
        c3 = self.stability_carry(returns)

        idx = c1.index.intersection(c2.index).intersection(c3.index)
        combined = (
            c1.reindex(idx) + c2.reindex(idx) + c3.reindex(idx)
        ) / 3.0

        return self._normalise(combined).dropna(how="all")

    # ------------------------------------------------------------------
    # Helper
    # ------------------------------------------------------------------

    @staticmethod
    def _normalise(df: pd.DataFrame) -> pd.DataFrame:
        mu  = df.mean(axis=1)
        std = df.std(axis=1)
        return df.sub(mu, axis=0).div(std + 1e-8, axis=0)
