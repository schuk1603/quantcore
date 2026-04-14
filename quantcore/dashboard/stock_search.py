"""
Stock search and valuation engine.

Pulls real-time data from Yahoo Finance and news from Yahoo Finance RSS
and New York Times RSS (no API key required).
"""

import numpy as np
import pandas as pd
import yfinance as yf
import urllib.request
import xml.etree.ElementTree as ET
from typing import Dict, List, Optional
from datetime import datetime


# ── News fetchers ──────────────────────────────────────────────────

def fetch_yahoo_news(ticker: str, max_items: int = 6) -> List[Dict]:
    """Fetch latest news for a ticker from Yahoo Finance."""
    try:
        stock = yf.Ticker(ticker)
        news  = stock.news or []
        results = []
        for item in news[:max_items]:
            content = item.get("content", {})
            title   = content.get("title", item.get("title", "No title"))
            summary = content.get("summary", "")
            pub_date = content.get("pubDate", "")
            # Get URL from clickThroughUrl or canonicalUrl
            url = (content.get("clickThroughUrl", {}) or {}).get("url", "")
            if not url:
                url = (content.get("canonicalUrl", {}) or {}).get("url", "#")
            results.append({
                "source":  "Yahoo Finance",
                "title":   title,
                "summary": summary[:200] + "..." if len(summary) > 200 else summary,
                "url":     url,
                "date":    pub_date[:10] if pub_date else "",
            })
        return results
    except Exception:
        return []


def fetch_nyt_news(query: str, max_items: int = 4) -> List[Dict]:
    """
    Fetch NYT headlines via their public RSS feed (no API key needed).
    Searches the business/markets feed and filters by query keyword.
    """
    feeds = [
        "https://rss.nytimes.com/services/xml/rss/nyt/Business.xml",
        "https://rss.nytimes.com/services/xml/rss/nyt/Economy.xml",
    ]
    results = []
    query_lower = query.lower()

    for feed_url in feeds:
        try:
            req = urllib.request.Request(
                feed_url,
                headers={"User-Agent": "Mozilla/5.0"}
            )
            with urllib.request.urlopen(req, timeout=5) as response:
                xml_data = response.read()
            root = ET.fromstring(xml_data)
            ns   = {"media": "http://search.yahoo.com/mrss/"}

            for item in root.findall(".//item"):
                title       = item.findtext("title", "")
                description = item.findtext("description", "")
                link        = item.findtext("link", "#")
                pub_date    = item.findtext("pubDate", "")

                # Filter by relevance to query
                combined = (title + " " + description).lower()
                if query_lower in combined or any(
                    word in combined for word in query_lower.split()
                ):
                    results.append({
                        "source":  "New York Times",
                        "title":   title,
                        "summary": description[:200] + "..." if len(description) > 200 else description,
                        "url":     link,
                        "date":    pub_date[:16] if pub_date else "",
                    })
                    if len(results) >= max_items:
                        return results
        except Exception:
            continue

    return results


# ── Stock data ─────────────────────────────────────────────────────

def get_stock_data(ticker: str) -> Dict:
    """
    Pull comprehensive stock data from Yahoo Finance.

    Returns a dict with:
      info         : key financial metrics
      price_history: 1-year price series
      news         : combined Yahoo + NYT news
      valuation    : buy/hold/sell signal with reasoning
    """
    try:
        stock = yf.Ticker(ticker.upper())
        info  = stock.info or {}

        # ── Key metrics ────────────────────────────────────────────
        def safe(key, default="N/A"):
            v = info.get(key, default)
            return default if v is None else v

        current_price   = safe("currentPrice", safe("regularMarketPrice", 0))
        prev_close      = safe("previousClose", 0)
        price_change    = (current_price - prev_close) if isinstance(current_price, (int, float)) and isinstance(prev_close, (int, float)) else 0
        price_change_pct = (price_change / prev_close * 100) if prev_close else 0

        metrics = {
            "Name":              safe("longName", safe("shortName", ticker)),
            "Ticker":            ticker.upper(),
            "Current Price":     f"${current_price:,.2f}" if isinstance(current_price, (int, float)) else "N/A",
            "Change":            f"{price_change:+.2f} ({price_change_pct:+.2f}%)" if isinstance(price_change, (int, float)) else "N/A",
            "Market Cap":        _fmt_large(safe("marketCap")),
            "52W High":          f"${safe('fiftyTwoWeekHigh'):,.2f}" if isinstance(safe('fiftyTwoWeekHigh'), (int, float)) else "N/A",
            "52W Low":           f"${safe('fiftyTwoWeekLow'):,.2f}"  if isinstance(safe('fiftyTwoWeekLow'),  (int, float)) else "N/A",
            "P/E Ratio":         f"{safe('trailingPE'):.1f}x"        if isinstance(safe('trailingPE'),       (int, float)) else "N/A",
            "Forward P/E":       f"{safe('forwardPE'):.1f}x"         if isinstance(safe('forwardPE'),        (int, float)) else "N/A",
            "PEG Ratio":         f"{safe('pegRatio'):.2f}"            if isinstance(safe('pegRatio'),         (int, float)) else "N/A",
            "Price/Book":        f"{safe('priceToBook'):.2f}x"        if isinstance(safe('priceToBook'),      (int, float)) else "N/A",
            "Revenue (TTM)":     _fmt_large(safe("totalRevenue")),
            "Revenue Growth":    f"{safe('revenueGrowth', 0)*100:.1f}%" if isinstance(safe('revenueGrowth'), (int, float)) else "N/A",
            "Gross Margin":      f"{safe('grossMargins', 0)*100:.1f}%"  if isinstance(safe('grossMargins'),  (int, float)) else "N/A",
            "Net Margin":        f"{safe('profitMargins', 0)*100:.1f}%" if isinstance(safe('profitMargins'), (int, float)) else "N/A",
            "EPS (TTM)":         f"${safe('trailingEps'):.2f}"          if isinstance(safe('trailingEps'),   (int, float)) else "N/A",
            "Dividend Yield":    f"{safe('dividendYield', 0)*100:.2f}%" if isinstance(safe('dividendYield'), (int, float)) else "N/A",
            "Beta":              f"{safe('beta'):.2f}"                  if isinstance(safe('beta'),           (int, float)) else "N/A",
            "Analyst Rating":    safe("recommendationKey", "N/A").replace("_", " ").title(),
            "Target Price":      f"${safe('targetMeanPrice'):,.2f}"     if isinstance(safe('targetMeanPrice'), (int, float)) else "N/A",
            "Sector":            safe("sector", "N/A"),
            "Industry":          safe("industry", "N/A"),
        }

        # ── Price history (1 year) ─────────────────────────────────
        hist = stock.history(period="1y")
        price_history = hist["Close"] if not hist.empty else pd.Series(dtype=float)

        # ── News ──────────────────────────────────────────────────
        company_name = info.get("longName", ticker)
        yahoo_news   = fetch_yahoo_news(ticker)
        nyt_news     = fetch_nyt_news(company_name)
        all_news     = yahoo_news + nyt_news

        # ── Valuation signal ──────────────────────────────────────
        valuation = _compute_valuation(info, price_history)

        return {
            "ticker":        ticker.upper(),
            "metrics":       metrics,
            "price_history": price_history,
            "news":          all_news,
            "valuation":     valuation,
            "error":         None,
        }

    except Exception as e:
        return {"ticker": ticker, "metrics": {}, "price_history": pd.Series(dtype=float),
                "news": [], "valuation": {}, "error": str(e)}


def _compute_valuation(info: Dict, price_history: pd.Series) -> Dict:
    """
    Generate a simple multi-factor valuation signal.

    Scoring rubric (each factor contributes to a score):
      Momentum   : 1-year price return vs S&P 500 proxy
      Value      : P/E vs sector average
      Quality    : Profit margin, revenue growth
      Risk       : Beta, volatility
    """
    score   = 0
    reasons = []
    signals = {}

    # ── Momentum ──────────────────────────────────────────────────
    if not price_history.empty and len(price_history) > 20:
        one_yr_ret = (price_history.iloc[-1] / price_history.iloc[0] - 1) * 100
        signals["1Y Return"] = f"{one_yr_ret:.1f}%"
        if one_yr_ret > 20:
            score += 2
            reasons.append(f"Strong 1-year return of {one_yr_ret:.1f}%")
        elif one_yr_ret > 0:
            score += 1
            reasons.append(f"Positive 1-year return of {one_yr_ret:.1f}%")
        else:
            score -= 1
            reasons.append(f"Negative 1-year return of {one_yr_ret:.1f}%")

        # Volatility
        daily_vol = price_history.pct_change().std() * np.sqrt(252) * 100
        signals["Annualised Vol"] = f"{daily_vol:.1f}%"
        if daily_vol < 20:
            score += 1
            reasons.append("Low volatility — stable price action")
        elif daily_vol > 50:
            score -= 1
            reasons.append("High volatility — risky price action")

    # ── Value ─────────────────────────────────────────────────────
    pe = info.get("trailingPE")
    if pe and isinstance(pe, (int, float)):
        signals["P/E"] = f"{pe:.1f}x"
        if pe < 15:
            score += 2
            reasons.append(f"Attractive P/E of {pe:.1f}x — potentially undervalued")
        elif pe < 25:
            score += 1
            reasons.append(f"Reasonable P/E of {pe:.1f}x")
        elif pe > 40:
            score -= 1
            reasons.append(f"Elevated P/E of {pe:.1f}x — priced for perfection")

    peg = info.get("pegRatio")
    if peg and isinstance(peg, (int, float)):
        if peg < 1:
            score += 2
            reasons.append(f"PEG ratio {peg:.2f} < 1 — growth at reasonable price")
        elif peg > 2:
            score -= 1
            reasons.append(f"PEG ratio {peg:.2f} — expensive relative to growth")

    # ── Quality ───────────────────────────────────────────────────
    margin = info.get("profitMargins")
    if margin and isinstance(margin, (int, float)):
        signals["Net Margin"] = f"{margin*100:.1f}%"
        if margin > 0.20:
            score += 2
            reasons.append(f"High net margin of {margin*100:.1f}%")
        elif margin > 0.10:
            score += 1
            reasons.append(f"Solid net margin of {margin*100:.1f}%")
        elif margin < 0:
            score -= 2
            reasons.append(f"Negative net margin — company losing money")

    rev_growth = info.get("revenueGrowth")
    if rev_growth and isinstance(rev_growth, (int, float)):
        signals["Revenue Growth"] = f"{rev_growth*100:.1f}%"
        if rev_growth > 0.20:
            score += 2
            reasons.append(f"Strong revenue growth of {rev_growth*100:.1f}%")
        elif rev_growth > 0.05:
            score += 1
            reasons.append(f"Moderate revenue growth of {rev_growth*100:.1f}%")
        elif rev_growth < 0:
            score -= 1
            reasons.append(f"Declining revenue — {rev_growth*100:.1f}%")

    # ── Analyst consensus ─────────────────────────────────────────
    rec = info.get("recommendationKey", "")
    if rec in ("strong_buy", "buy"):
        score += 2
        reasons.append(f"Analysts rate it: {rec.replace('_',' ').title()}")
    elif rec == "hold":
        reasons.append("Analysts rate it: Hold")
    elif rec in ("sell", "strong_sell"):
        score -= 2
        reasons.append(f"Analysts rate it: {rec.replace('_',' ').title()}")

    # ── Final verdict ─────────────────────────────────────────────
    if score >= 6:
        verdict = "STRONG BUY"
        color   = "#26a69a"
    elif score >= 3:
        verdict = "BUY"
        color   = "#66bb6a"
    elif score >= 0:
        verdict = "HOLD"
        color   = "#ffa726"
    elif score >= -3:
        verdict = "SELL"
        color   = "#ef5350"
    else:
        verdict = "STRONG SELL"
        color   = "#b71c1c"

    return {
        "verdict": verdict,
        "color":   color,
        "score":   score,
        "reasons": reasons,
        "signals": signals,
    }


def _fmt_large(val) -> str:
    """Format large numbers as $1.2T, $345B, $12M."""
    if not isinstance(val, (int, float)) or val == "N/A":
        return "N/A"
    if val >= 1e12:
        return f"${val/1e12:.2f}T"
    if val >= 1e9:
        return f"${val/1e9:.2f}B"
    if val >= 1e6:
        return f"${val/1e6:.2f}M"
    return f"${val:,.0f}"
