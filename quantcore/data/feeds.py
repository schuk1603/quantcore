"""
Market data feed with local SQLite caching and multi-source support.
"""

import sqlite3
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import yfinance as yf

CACHE_DIR = Path.home() / ".quantcore" / "cache"


class DataFeed:
    """
    Downloads, caches, and serves OHLCV market data.

    Uses yfinance as the primary source with a local SQLite cache to
    avoid redundant network calls. Supports adjusted and unadjusted prices.
    """

    def __init__(self, cache_dir: Optional[Path] = None):
        self.cache_dir = cache_dir or CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._db_path = self.cache_dir / "market_data.db"
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self._db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS ohlcv (
                    ticker   TEXT,
                    date     TEXT,
                    open     REAL,
                    high     REAL,
                    low      REAL,
                    close    REAL,
                    volume   REAL,
                    PRIMARY KEY (ticker, date)
                )
            """)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_prices(
        self,
        tickers: List[str],
        start: str,
        end: str,
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """Return adjusted close prices as a DataFrame indexed by date."""
        frames: dict[str, pd.Series] = {}
        to_download: List[str] = []

        for ticker in tickers:
            if use_cache:
                cached = self._load_from_cache(ticker, start, end)
                if cached is not None and not cached.empty:
                    frames[ticker] = cached
                    continue
            to_download.append(ticker)

        if to_download:
            raw = yf.download(
                to_download, start=start, end=end,
                auto_adjust=True, progress=False, threads=True,
            )
            # yfinance returns MultiIndex only when len(tickers) > 1
            if len(to_download) == 1:
                close = raw[["Close"]].rename(columns={"Close": to_download[0]})
            else:
                close = raw["Close"]

            for ticker in to_download:
                if ticker in close.columns:
                    series = close[ticker].dropna()
                    self._save_to_cache(ticker, series)
                    frames[ticker] = series

        prices = pd.DataFrame(frames)
        prices.index = pd.to_datetime(prices.index)
        return prices.sort_index().dropna(how="all")

    def get_returns(
        self,
        prices: pd.DataFrame,
        log: bool = True,
    ) -> pd.DataFrame:
        """Compute log or simple returns from a price DataFrame."""
        if log:
            return np.log(prices / prices.shift(1)).dropna()
        return prices.pct_change().dropna()

    def get_ohlcv(self, ticker: str, start: str, end: str) -> pd.DataFrame:
        """Return full OHLCV DataFrame for a single ticker."""
        raw = yf.download(
            ticker, start=start, end=end,
            auto_adjust=True, progress=False,
        )
        raw.index = pd.to_datetime(raw.index)
        return raw.sort_index()

    # ------------------------------------------------------------------
    # Cache helpers
    # ------------------------------------------------------------------

    def _load_from_cache(
        self, ticker: str, start: str, end: str
    ) -> Optional[pd.Series]:
        try:
            with sqlite3.connect(self._db_path) as conn:
                df = pd.read_sql(
                    "SELECT date, close FROM ohlcv "
                    "WHERE ticker=? AND date>=? AND date<=? ORDER BY date",
                    conn,
                    params=(ticker, start, end),
                )
            if df.empty:
                return None
            df["date"] = pd.to_datetime(df["date"])
            return df.set_index("date")["close"]
        except Exception:
            return None

    def _save_to_cache(self, ticker: str, series: pd.Series):
        try:
            with sqlite3.connect(self._db_path) as conn:
                for date, price in series.items():
                    conn.execute(
                        "INSERT OR REPLACE INTO ohlcv (ticker,date,close) VALUES (?,?,?)",
                        (ticker, str(date.date()), float(price)),
                    )
        except Exception:
            pass

    def clear_cache(self):
        """Wipe the local cache database."""
        if self._db_path.exists():
            self._db_path.unlink()
        self._init_db()
