"""
AI Analyst for QuantCore — powered by Google Gemini (free tier).

Google Gemini API is completely free with no credit card required.
Free tier: 15 requests/minute, 1,500 requests/day.

Get your free key at: https://aistudio.google.com/app/apikey
Then set: GEMINI_API_KEY=your_key_here
"""

import os
import pandas as pd
from typing import Dict

# ── Attempt to load Gemini SDK ─────────────────────────────────────
GEMINI_AVAILABLE = False
_client = None

try:
    import google.generativeai as genai
    _api_key = os.environ.get("GEMINI_API_KEY", "")
    if _api_key:
        genai.configure(api_key=_api_key)
        _client = genai.GenerativeModel("gemini-1.5-flash")
        GEMINI_AVAILABLE = True
except ImportError:
    pass


def _ask(prompt: str, fallback: str = "") -> str:
    """Send a prompt to Gemini; return fallback on any error."""
    if _client is None:
        return fallback
    try:
        response = _client.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"[Oracle error: {e}]"


# ── Public API ─────────────────────────────────────────────────────

def portfolio_insight(
    tearsheet: pd.DataFrame,
    risk_report: Dict,
    regime_stats=None,
) -> str:
    """Auto-generated AI narrative about the portfolio's performance."""
    if not GEMINI_AVAILABLE:
        return (
            "The Oracle sleeps. Set the GEMINI_API_KEY environment variable "
            "to awaken AI-powered portfolio counsel. "
            "Free key (no credit card): https://aistudio.google.com/app/apikey"
        )

    ts_dict = (
        tearsheet["Value"].to_dict()
        if "Value" in tearsheet.columns
        else tearsheet.to_dict()
    )
    regime_str = (
        regime_stats.to_string()
        if regime_stats is not None and not regime_stats.empty
        else "N/A"
    )

    prompt = f"""You are the Chief Quantitative Strategist at QuantCore — an elite,
Viking-themed institutional hedge fund. Speak with authority and precision.

Portfolio Performance: {ts_dict}
Risk Report: {risk_report}
Market Regime Statistics: {regime_str}

Write a 5-6 sentence professional portfolio analysis. Lead with the most important
quantitative finding (CAGR, Sharpe, drawdown). Comment on regime exposure.
Use 1 brief Viking/Norse metaphor naturally. End with a clear strategic recommendation.
Plain text only — no bullet points, no markdown. Under 150 words. Be direct."""

    return _ask(
        prompt,
        fallback="Portfolio analysis unavailable. GEMINI_API_KEY not configured.",
    )


def stock_ai_commentary(ticker: str, metrics: Dict, valuation: Dict) -> str:
    """Concise AI analyst note for a stock. Empty string if API unavailable."""
    if not GEMINI_AVAILABLE:
        return ""

    verdict = valuation.get("verdict", "HOLD")
    score   = valuation.get("score", 0)
    reasons = valuation.get("reasons", [])
    key_m   = {
        k: v for k, v in metrics.items()
        if k in ("Current Price", "P/E Ratio", "Forward P/E", "PEG Ratio",
                 "Revenue Growth", "Net Margin", "Beta", "Analyst Rating",
                 "Target Price", "Market Cap")
    }

    prompt = f"""You are a senior equity analyst at QuantCore hedge fund.
Write a sharp, data-driven analyst note on {ticker}.

Quant Signal: {verdict} (score: {score}/12)
Key Metrics: {key_m}
Signal Drivers: {reasons}

3 sentences maximum. Lead with the dominant factor. Cite specific numbers.
Professional hedge-fund tone. No markdown. Under 70 words."""

    return _ask(prompt, fallback="")


def ask_odin(question: str, tearsheet_str: str, risk_str: str) -> str:
    """Answer a user question about the portfolio via Gemini AI."""
    if not GEMINI_AVAILABLE:
        return (
            "Odin's Oracle is offline. Set the GEMINI_API_KEY environment variable "
            "to activate AI counsel.\n\n"
            "Free key (no credit card needed): https://aistudio.google.com/app/apikey"
        )

    prompt = f"""You are Odin — the all-knowing AI war counsel embedded in QuantCore,
an elite quantitative research platform.

Portfolio Performance Context: {tearsheet_str}
Risk Context: {risk_str}

Warrior's Question: {question}

Answer as a veteran quantitative strategist. Be specific and cite actual portfolio
statistics where relevant. Use no more than 1 Norse warrior metaphor.
Answer in 3-5 sentences max. Professional, direct, data-driven.
Plain text only — no markdown or bullet points."""

    return _ask(
        prompt,
        fallback="The Oracle could not be reached. Check your GEMINI_API_KEY.",
    )
