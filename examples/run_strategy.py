"""
QuantCore — Full strategy example with 3D dashboard and regime detection.

Usage:
    cd C:\\Users\\samue\\Downloads\\quantcore
    python examples/run_strategy.py
Then open: http://127.0.0.1:8050
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd

from quantcore.data.feeds            import DataFeed
from quantcore.signals.momentum      import MomentumSignal
from quantcore.signals.mean_reversion import MeanReversionSignal
from quantcore.signals.factors       import FactorModel
from quantcore.signals.regime        import RegimeDetector
from quantcore.signals.decay         import FactorDecayAnalyzer
from quantcore.signals.trend         import TrendFollowingSignal
from quantcore.signals.carry         import CarrySignal
from quantcore.portfolio.optimizer   import PortfolioOptimizer
from quantcore.portfolio.sizing      import KellyCriterion
from quantcore.risk.engine           import RiskEngine
from quantcore.backtest.engine       import BacktestEngine, CostModel
from quantcore.analytics.performance import PerformanceAnalytics
from quantcore.dashboard.app         import build_dashboard

# ── Config ─────────────────────────────────────────────────────────
UNIVERSE = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META",
    "NVDA", "TSLA", "JPM",  "JNJ",  "V",
    "BRK-B","UNH",  "XOM",  "PG",   "HD",
]
BENCHMARK   = "SPY"
START       = "2018-01-01"
END         = "2024-12-31"
INITIAL_CAP = 100_000
TOP_N       = 6
LONG_ONLY   = True
REBAL_FREQ  = "ME"

# ── 1. Data ─────────────────────────────────────────────────────────
print("Downloading market data...")
feed         = DataFeed()
prices       = feed.get_prices(UNIVERSE + [BENCHMARK], START, END)
asset_prices = prices[UNIVERSE].dropna(how="all")
bm_prices    = prices[[BENCHMARK]].dropna()
returns      = feed.get_returns(asset_prices, log=True)
bm_returns   = feed.get_returns(bm_prices,    log=True)[BENCHMARK]
print(f"Loaded {len(asset_prices)} days × {len(UNIVERSE)} assets")

# ── 2. Signal function ──────────────────────────────────────────────
mom_signal   = MomentumSignal(lookback=252, skip=21)
mr_signal    = MeanReversionSignal(window=63)
factor_mdl   = FactorModel()
trend_signal = TrendFollowingSignal(fast_span=8, slow_span=24)
carry_signal = CarrySignal(window=252)

def compute_alpha(prices_td: pd.DataFrame, date: pd.Timestamp) -> pd.Series:
    rets = np.log(prices_td / prices_td.shift(1)).dropna()
    signals = []
    for sig_fn in [
        lambda: mom_signal.combined(prices_td, rets),
        lambda: mr_signal.combined(prices_td),
        lambda: factor_mdl.composite_alpha(prices_td, rets),
        lambda: trend_signal.combined(prices_td, rets),
        lambda: carry_signal.combined(prices_td, rets),
    ]:
        try:
            s = sig_fn()
            if date in s.index:
                signals.append(s.loc[date])
        except Exception:
            pass
    if not signals:
        return pd.Series(dtype=float)

    combined = pd.concat(signals, axis=1).mean(axis=1).dropna()
    top      = combined.nlargest(TOP_N).clip(lower=0)
    sel_rets = rets[top.index].dropna()
    if len(sel_rets) < 30 or sel_rets.shape[1] < 2:
        return pd.Series(1.0 / len(top), index=top.index)
    try:
        w = PortfolioOptimizer(sel_rets).max_sharpe(long_only=LONG_ONLY)
    except Exception:
        w = pd.Series(1.0 / len(top), index=top.index)
    w = w.clip(lower=0, upper=0.4)
    return w / w.abs().sum()

# ── 3. Backtest ─────────────────────────────────────────────────────
print("Running backtest...")
engine = BacktestEngine(
    prices          = asset_prices,
    signal_fn       = compute_alpha,
    initial_capital = INITIAL_CAP,
    cost_model      = CostModel(commission_fixed=1.0, commission_pct=0.001,
                                bid_ask_spread=0.0005, market_impact=0.10),
    rebalance_freq  = REBAL_FREQ,
    benchmark       = bm_prices[BENCHMARK] * (INITIAL_CAP / bm_prices[BENCHMARK].iloc[0]),
)
result = engine.run()
print(f"Backtest complete. Final equity: ${result['equity'].iloc[-1]:,.0f}")

# ── 4. Performance ──────────────────────────────────────────────────
port_returns = result["returns"]
perf = PerformanceAnalytics(port_returns, benchmark=bm_returns, risk_free_rate=0.05)
tearsheet      = perf.tearsheet()
rolling_sharpe = perf.rolling_sharpe(252)
rolling_vol    = perf.rolling_volatility(63)
print("\n" + "=" * 50)
print(tearsheet.to_string())
print("=" * 50)

# ── 5. Risk ─────────────────────────────────────────────────────────
risk_engine = RiskEngine(port_returns)
risk_report = risk_engine.full_report()
print("\nRisk Report:")
for k, v in risk_report.items():
    print(f"  {k:<30}: {v}")

# ── 6. Efficient Frontier ────────────────────────────────────────────
print("\nComputing efficient frontier...")
frontier = pd.DataFrame()
try:
    last_252 = returns.iloc[-252:].dropna(how="any", axis=1)
    frontier = PortfolioOptimizer(last_252, risk_free_rate=0.05).efficient_frontier(40)
except Exception as e:
    print(f"Frontier skipped: {e}")

# ── 7. Regime Detection (HMM) ───────────────────────────────────────
print("Fitting regime model...")
regime_labels = None
regime_probs  = None
regime_stats  = None
try:
    detector      = RegimeDetector(n_states=3, max_iter=150)
    detector.fit(port_returns)
    regime_labels = detector.predict(port_returns)
    regime_probs  = detector.regime_probs(port_returns)
    regime_stats  = detector.regime_stats(port_returns, regime_labels)
    print("Regime stats:")
    print(regime_stats.to_string())
except Exception as e:
    print(f"Regime detection skipped: {e}")

# ── 8. Factor Decay ─────────────────────────────────────────────────
print("Computing factor decay...")
ic_decay  = pd.DataFrame()
ic_surface = pd.DataFrame()
try:
    analyzer  = FactorDecayAnalyzer(max_horizon=41)
    # Use momentum signal scores for decay analysis
    mom_scores = mom_signal.cross_sectional(asset_prices).dropna(how="all")
    # Sample to speed up computation
    sample_prices = asset_prices.iloc[::5]
    sample_scores = mom_scores.reindex(sample_prices.index)
    ic_decay   = analyzer.ic_decay(sample_scores, sample_prices)
    print(f"IC decay computed over {len(ic_decay)} horizons. "
          f"Half-life ≈ {analyzer.half_life(ic_decay):.0f} days")
    ic_surface = analyzer.rolling_ic_surface(
        sample_scores, sample_prices,
        horizons=[1, 3, 5, 10, 15, 21], window=40,
    )
except Exception as e:
    print(f"Factor decay skipped: {e}")

# ── 9. Kelly Sizing ─────────────────────────────────────────────────
kelly         = KellyCriterion(fraction=0.5)
kelly_weights = kelly.multi_asset(returns.iloc[-252:].dropna(how="any", axis=1))
print(f"\nHalf-Kelly weights:\n{kelly_weights.round(3)}")

# ── 10. Dashboard ───────────────────────────────────────────────────
import traceback

print("\nBuilding dashboard...")
try:
    app = build_dashboard(
        backtest_result    = result,
        tearsheet          = tearsheet,
        risk_report        = risk_report,
        rolling_sharpe     = rolling_sharpe,
        rolling_vol        = rolling_vol,
        efficient_frontier = frontier,
        regime_labels      = regime_labels,
        regime_probs       = regime_probs,
        regime_stats       = regime_stats,
        ic_decay           = ic_decay,
        ic_surface         = ic_surface,
        asset_returns      = returns,
        asset_prices       = asset_prices,
    )
    print("Dashboard built OK.")
except Exception:
    print("\n--- BUILD ERROR ---")
    traceback.print_exc()
    raise SystemExit(1)

import os
port = int(os.environ.get("PORT", 8051))
host = "0.0.0.0"
print(f"Launching at http://{host}:{port} ...")
try:
    from waitress import serve
    serve(app.server, host=host, port=port)
except Exception:
    print("\n--- SERVER ERROR ---")
    traceback.print_exc()
