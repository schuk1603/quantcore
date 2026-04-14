"""
Gemini AI Analyst for QuantCore.

Uses Google's Gemini 2.0 Flash model to provide AI-powered portfolio
insights, stock commentary, and an interactive "Ask Odin" war counsel.

Set the GEMINI_API_KEY environment variable to enable AI features.
Free API key: https://aistudio.google.com/app/apikey
"""

import os
import pandas as pd
from typing import Dict, Optional

# ── Attempt to load Gemini ─────────────────────────────────────────
GEMINI_AVAILABLE = False
_genai = None

try:
    import google.generativeai as _genai_module
    _api_key = os.environ.get("GEMINI_API_KEY", "")
    if _api_key:
        _genai_module.configure(api_key=_api_key)
        _genai = _genai_module
        GEMINI_AVAILABLE = True
except ImportError:
    pass


def _model():
    """Return a configured Gemini model instance."""
    if _genai is None:
        return None
    for model_id in ("gemini-2.0-flash", "gemini-1.5-flash", "gemini-pro"):
        try:
            return _genai.GenerativeModel(model_id)
        except Exception:
            continue
    return None


def _safe_generate(prompt: str, fallback: str = "") -> str:
    """Call Gemini safely; return fallback on any error."""
    m = _model()
    if m is None:
        return fallback
    try:
        response = m.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"[Oracle error: {e}]"


# ── Public API ─────────────────────────────────────────────────────

def portfolio_insight(
    tearsheet: pd.DataFrame,
    risk_report: Dict,
    regime_stats=None,
) -> str:
    """
    Generate a professional AI narrative about the portfolio's performance.
    Returned as plain text (HTML-safe, no markdown syntax).
    """
    if not GEMINI_AVAILABLE:
        return (
            "The Oracle sleeps. Set the GEMINI_API_KEY environment variable "
            "to awaken AI-powered portfolio counsel from Odin's war advisors."
        )

    ts_dict = (
        tearsheet["Value"].to_dict()
        if "Value" in tearsheet.columns
        else tearsheet.to_dict()
    )
    regime_str = (
        regime_stats.to_string() if regime_stats is not None and not regime_stats.empty
        else "N/A"
    )

    prompt = f"""You are the Chief Quantitative Strategist at QuantCore — an elite, Viking-themed
institutional hedge fund. Speak with authority, precision, and a faint Norse gravitas.

Portfolio Performance Metrics:
{ts_dict}

Risk Report:
{risk_report}

Market Regime Statistics:
{regime_str}

Write a 5-7 sentence professional portfolio analysis. Lead with the most important quantitative
finding (e.g., CAGR vs benchmark, Sharpe quality, drawdown risk). Comment on regime exposure.
Weave in 1-2 Viking/Norse metaphors naturally — warrior, Valhalla, forge, etc.
End with a clear strategic recommendation (scale up, tighten risk, rebalance).
Plain text only — no bullet points, no markdown. Under 160 words. Be direct."""

    return _safe_generate(
        prompt,
        fallback="Portfolio analysis unavailable. Gemini API key not configured.",
    )


def stock_ai_commentary(ticker: str, metrics: Dict, valuation: Dict) -> str:
    """
    Generate a concise AI equity analyst note for a given stock.
    Returns empty string if Gemini is unavailable (feature degrades gracefully).
    """
    if not GEMINI_AVAILABLE:
        return ""

    verdict  = valuation.get("verdict", "HOLD")
    score    = valuation.get("score", 0)
    reasons  = valuation.get("reasons", [])
    key_m    = {
        k: v for k, v in metrics.items()
        if k in ("Current Price", "P/E Ratio", "Forward P/E", "PEG Ratio",
                 "Revenue Growth", "Net Margin", "Beta", "Analyst Rating",
                 "Target Price", "Market Cap")
    }

    prompt = f"""You are a senior equity analyst at QuantCore hedge fund. Write a sharp,
data-driven analyst note on {ticker}.

Quant Signal: {verdict} (composite score: {score}/12)
Key Metrics: {key_m}
Signal Drivers: {reasons}

Write 3 sentences maximum. Lead with the dominant factor. Cite specific numbers.
Professional hedge-fund tone. No markdown. Under 70 words."""

    return _safe_generate(prompt, fallback="")


def ask_odin(question: str, tearsheet_str: str, risk_str: str) -> str:
    """
    Answer a user question about the portfolio via Gemini.
    This is the interactive 'Ask Odin' oracle in the dashboard.
    """
    if not GEMINI_AVAILABLE:
        return (
            "Odin's Oracle is offline. Add the GEMINI_API_KEY environment variable "
            "to your system or Railway/Render settings to awaken AI counsel.\n\n"
            "Free key at: https://aistudio.google.com/app/apikey"
        )

    prompt = f"""You are Odin — the all-knowing AI war counsel embedded in QuantCore,
an elite quantitative research platform.

Portfolio Context (Performance): {tearsheet_str}
Portfolio Context (Risk):        {risk_str}

Warrior's Question: {question}

Answer as a veteran quantitative strategist. Be specific and cite actual portfolio statistics
where relevant. Use no more than 1 Norse warrior metaphor. Answer in 3-5 sentences max.
Professional, direct, data-driven. Plain text only — no markdown or bullet points."""

    return _safe_generate(
        prompt,
        fallback="The Oracle could not be reached. Check your GEMINI_API_KEY.",
    )
