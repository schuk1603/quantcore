"""
Event-driven backtesting engine.

Features:
  - Realistic transaction costs (fixed + proportional + market impact)
  - Bid-ask spread slippage
  - Portfolio rebalancing on signal
  - Short-selling support
  - Walk-forward validation
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple


# ------------------------------------------------------------------
# Data structures
# ------------------------------------------------------------------

@dataclass
class Order:
    date:    pd.Timestamp
    ticker:  str
    shares:  float          # positive = buy, negative = sell
    price:   float
    cost:    float = 0.0    # transaction cost incurred


@dataclass
class Portfolio:
    initial_capital: float = 100_000.0
    cash:            float = field(init=False)
    positions:       Dict[str, float] = field(default_factory=dict)  # ticker → shares
    trades:          List[Order]      = field(default_factory=list)
    equity_curve:    List[Tuple]      = field(default_factory=list)   # (date, equity)

    def __post_init__(self):
        self.cash = self.initial_capital

    def market_value(self, prices: Dict[str, float]) -> float:
        pos_value = sum(
            shares * prices.get(ticker, 0)
            for ticker, shares in self.positions.items()
        )
        return self.cash + pos_value

    def weights(self, prices: Dict[str, float]) -> Dict[str, float]:
        mv = self.market_value(prices)
        if mv <= 0:
            return {}
        return {
            t: (s * prices.get(t, 0)) / mv
            for t, s in self.positions.items()
        }


# ------------------------------------------------------------------
# Transaction cost model
# ------------------------------------------------------------------

@dataclass
class CostModel:
    """
    Realistic transaction cost model.

    Parameters
    ----------
    commission_fixed : float
        Fixed commission per trade in dollars (default $1.00).
    commission_pct : float
        Proportional commission as fraction of trade value (default 0.001 = 10 bps).
    bid_ask_spread : float
        Half bid-ask spread as fraction of price (default 0.0005 = 5 bps).
    market_impact : float
        Market impact coefficient (default 0.1 — Almgren-Chriss style).
    daily_volume_frac : float
        Max fraction of daily volume per trade for impact calculation.
    """
    commission_fixed:   float = 1.00
    commission_pct:     float = 0.001
    bid_ask_spread:     float = 0.0005
    market_impact:      float = 0.10
    daily_volume_frac:  float = 0.01

    def total_cost(
        self,
        shares: float,
        price:  float,
        daily_volume: float = 1e6,
    ) -> float:
        """
        Compute total transaction cost for a trade.

        Cost = fixed commission + proportional commission
             + bid-ask spread + market impact
        """
        trade_value = abs(shares) * price

        commission  = self.commission_fixed + self.commission_pct * trade_value
        spread_cost = self.bid_ask_spread * trade_value
        vol_fraction = abs(shares) / (daily_volume * self.daily_volume_frac + 1)
        impact       = self.market_impact * vol_fraction * trade_value

        return commission + spread_cost + impact


# ------------------------------------------------------------------
# Main engine
# ------------------------------------------------------------------

class BacktestEngine:
    """
    Event-driven portfolio backtesting engine.

    Parameters
    ----------
    prices : pd.DataFrame
        Daily adjusted close prices (rows = dates, columns = tickers).
    signal_fn : callable
        Function that takes (prices_up_to_date, date) and returns a
        pd.Series of target weights indexed by ticker.
    initial_capital : float
        Starting capital in dollars.
    cost_model : CostModel, optional
        Transaction cost model (default: realistic costs).
    rebalance_freq : str
        Pandas offset alias for rebalancing (e.g. 'ME' = monthly end,
        'W' = weekly, 'B' = daily). Default 'ME'.
    """

    def __init__(
        self,
        prices: pd.DataFrame,
        signal_fn: Callable[[pd.DataFrame, pd.Timestamp], pd.Series],
        initial_capital: float = 100_000.0,
        cost_model: Optional[CostModel] = None,
        rebalance_freq: str = "M",
        benchmark: Optional[pd.Series] = None,
    ):
        self.prices          = prices.sort_index()
        self.signal_fn       = signal_fn
        self.initial_capital = initial_capital
        self.cost_model      = cost_model or CostModel()
        self.rebalance_freq  = rebalance_freq
        self.benchmark       = benchmark

    def run(self) -> pd.DataFrame:
        """
        Run the backtest and return a DataFrame of results indexed by date.

        Columns:
          equity, cash, returns, drawdown, turnover,
          benchmark_equity (if provided)
        """
        portfolio = Portfolio(initial_capital=self.initial_capital)
        dates     = self.prices.index
        rebal_dates = set(
            self.prices.resample(self.rebalance_freq).last().index
        )

        equity_curve  = []
        returns_list  = []
        turnover_list = []
        prev_equity   = self.initial_capital

        for i, date in enumerate(dates):
            current_prices = self.prices.loc[date].to_dict()
            equity = portfolio.market_value(current_prices)

            # Daily return
            daily_ret = (equity - prev_equity) / prev_equity if prev_equity > 0 else 0.0

            if date in rebal_dates and i >= 20:   # warm-up period
                prices_to_date = self.prices.iloc[: i + 1]
                try:
                    target_weights = self.signal_fn(prices_to_date, date)
                except Exception:
                    target_weights = pd.Series(dtype=float)

                turnover = self._rebalance(
                    portfolio, target_weights, current_prices, date
                )
            else:
                turnover = 0.0

            equity = portfolio.market_value(current_prices)
            equity_curve.append(equity)
            returns_list.append(daily_ret)
            turnover_list.append(turnover)
            prev_equity = equity

        equity_s   = pd.Series(equity_curve, index=dates)
        returns_s  = pd.Series(returns_list, index=dates)
        wealth     = (1 + returns_s).cumprod()
        peak       = wealth.cummax()
        drawdown_s = (wealth - peak) / peak

        result = pd.DataFrame({
            "equity":   equity_s,
            "returns":  returns_s,
            "drawdown": drawdown_s,
            "turnover": pd.Series(turnover_list, index=dates),
        })

        if self.benchmark is not None:
            bm = self.benchmark.reindex(dates).ffill()
            result["benchmark_equity"] = (
                self.initial_capital * (bm / bm.iloc[0])
            )

        daily_costs = (
            pd.Series(
                [o.cost for o in portfolio.trades],
                index=[o.date for o in portfolio.trades],
            )
            .groupby(level=0).sum()
            .reindex(dates)
            .fillna(0)
            .cumsum()
        )
        result["total_costs"] = daily_costs

        return result

    def _rebalance(
        self,
        portfolio: Portfolio,
        target_weights: pd.Series,
        current_prices: Dict[str, float],
        date: pd.Timestamp,
    ) -> float:
        """
        Rebalance portfolio to target weights.

        Returns total turnover fraction.
        """
        equity = portfolio.market_value(current_prices)
        if equity <= 0:
            return 0.0

        # Current weights
        current_weights = portfolio.weights(current_prices)
        tickers         = set(target_weights.index) | set(current_weights.keys())
        turnover        = 0.0

        for ticker in tickers:
            target_w  = float(target_weights.get(ticker, 0))
            current_w = current_weights.get(ticker, 0)
            price     = current_prices.get(ticker, 0)

            if price <= 0:
                continue

            delta_w = target_w - current_w
            if abs(delta_w) < 1e-4:
                continue

            delta_value  = delta_w * equity
            delta_shares = delta_value / price
            cost         = self.cost_model.total_cost(delta_shares, price)

            portfolio.cash -= delta_value + np.sign(delta_shares) * cost
            portfolio.positions[ticker] = portfolio.positions.get(ticker, 0) + delta_shares

            if abs(portfolio.positions[ticker]) < 1e-6:
                del portfolio.positions[ticker]

            portfolio.trades.append(
                Order(date=date, ticker=ticker,
                      shares=delta_shares, price=price, cost=cost)
            )

            turnover += abs(delta_w)

        return turnover

    # ------------------------------------------------------------------
    # Walk-forward validation
    # ------------------------------------------------------------------

    def walk_forward(
        self,
        train_periods: int = 252,
        test_periods:  int = 63,
    ) -> List[pd.DataFrame]:
        """
        Walk-forward (out-of-sample) validation.

        Splits the price history into overlapping train/test windows,
        runs a sub-backtest on each test window.

        Returns a list of per-window result DataFrames.
        """
        dates   = self.prices.index
        n       = len(dates)
        results = []
        start   = train_periods

        while start + test_periods <= n:
            test_prices = self.prices.iloc[start: start + test_periods]

            sub_engine = BacktestEngine(
                prices          = self.prices.iloc[: start + test_periods],
                signal_fn       = self.signal_fn,
                initial_capital = self.initial_capital,
                cost_model      = self.cost_model,
                rebalance_freq  = self.rebalance_freq,
            )
            result = sub_engine.run()
            results.append(result.iloc[start:])
            start += test_periods

        return results
