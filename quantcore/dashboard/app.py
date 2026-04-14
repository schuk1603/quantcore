"""
QuantCore — Viking-themed institutional quantitative research platform.

Powered by multi-factor alpha signals, HMM regime detection, trend following,
carry, ML signal combination, and Gemini AI counsel.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy.stats import norm

import dash
from dash import dcc, html, dash_table, Input, Output, State
import dash_bootstrap_components as dbc

from .stock_search    import get_stock_data
from .gemini_analyst  import (
    portfolio_insight, stock_ai_commentary, ask_odin, GEMINI_AVAILABLE
)

# ══════════════════════════════════════════════════════════════════
# VIKINGS COLOUR PALETTE
# ══════════════════════════════════════════════════════════════════
GOLD   = "#c9a227"   # hammered Norse gold
BLOOD  = "#8b0000"   # blood-stained iron
ICE    = "#87ceeb"   # fjord / Frost Giant blue
STEEL  = "#778899"   # tempered steel
FIRE   = "#e65100"   # forge ember
RUNE   = "#9e8a6d"   # carved rune stone
GREEN  = "#2e7d32"   # forest green
BG     = "#06060f"   # Niflheim abyss
CARD   = "#0c0c1e"   # dark iron plate
CARD2  = "#10102c"   # shield face
BORDER = "#1e1a3a"   # frame shadow
MIST   = "#999999"   # Bifrost mist

# ── Google Fonts: Cinzel (Norse/regal) + Rajdhani (body) ──────────
_FONTS = "https://fonts.googleapis.com/css2?family=Cinzel:wght@400;700;900&family=Rajdhani:wght@300;400;600;700&display=swap"
_FONT_H = "'Cinzel', serif"
_FONT_B = "'Rajdhani', sans-serif"


# ══════════════════════════════════════════════════════════════════
# STYLE HELPERS
# ══════════════════════════════════════════════════════════════════

def _card(children, border_color=GOLD, **kwargs):
    return dbc.Card(
        dbc.CardBody(children),
        style={
            "background":    CARD,
            "border":        f"1px solid {border_color}22",
            "borderTop":     f"2px solid {border_color}",
            "borderRadius":  "10px",
            "marginBottom":  "18px",
            "boxShadow":     f"0 0 18px {border_color}12",
        },
        **kwargs,
    )


def _desc(text: str):
    return html.P(text, style={
        "color":        MIST,
        "fontSize":     "0.82rem",
        "marginTop":    "-6px",
        "marginBottom": "12px",
        "fontStyle":    "italic",
        "fontFamily":   _FONT_B,
        "paddingLeft":  "4px",
        "letterSpacing": "0.3px",
    })


def _rune_header(rune: str, title: str, subtitle: str = "", color: str = GOLD):
    """
    Viking-styled section header with a rune glyph, Cinzel title font,
    and a glowing underline bar.
    """
    return dbc.Row(dbc.Col(html.Div([
        html.Div([
            html.Span(rune, style={
                "fontSize":   "1.6rem",
                "color":      color,
                "marginRight": "12px",
                "opacity":    "0.85",
                "fontFamily": _FONT_H,
                "verticalAlign": "middle",
            }),
            html.Span(title, style={
                "color":      color,
                "fontFamily": _FONT_H,
                "fontWeight": "700",
                "fontSize":   "1.15rem",
                "letterSpacing": "1.5px",
                "verticalAlign": "middle",
            }),
        ]),
        html.P(subtitle, style={
            "color":      MIST,
            "fontSize":   "0.8rem",
            "fontFamily": _FONT_B,
            "margin":     "2px 0 8px 38px",
            "letterSpacing": "0.5px",
        }) if subtitle else None,
        html.Hr(style={
            "borderColor": color,
            "opacity":     "0.25",
            "margin":      "6px 0 14px 0",
            "boxShadow":   f"0 1px 6px {color}55",
        }),
    ])))


def _kpi_stone(label: str, value: str, color: str = ICE, rune: str = "ᚠ"):
    """KPI rendered as a Norse runestone tablet."""
    return dbc.Col(_card([
        html.Div(rune, style={
            "color":      color,
            "opacity":    "0.4",
            "fontSize":   "1.2rem",
            "fontFamily": _FONT_H,
            "marginBottom": "2px",
        }),
        html.P(label, style={
            "color":         MIST,
            "fontSize":      "0.72rem",
            "marginBottom":  "3px",
            "textTransform": "uppercase",
            "letterSpacing": "1.5px",
            "fontFamily":    _FONT_B,
        }),
        html.H4(value, style={
            "color":      color,
            "fontWeight": "700",
            "margin":     0,
            "fontFamily": _FONT_H,
            "letterSpacing": "1px",
        }),
    ], border_color=color), xs=6, md=3)


def build_dashboard(
    backtest_result:    pd.DataFrame,
    tearsheet:          pd.DataFrame,
    risk_report:        dict,
    rolling_sharpe:     pd.Series,
    rolling_vol:        pd.Series,
    efficient_frontier: pd.DataFrame = None,
    regime_labels:      pd.Series    = None,
    regime_probs:       pd.DataFrame = None,
    regime_stats:       pd.DataFrame = None,
    ic_decay:           pd.DataFrame = None,
    ic_surface:         pd.DataFrame = None,
    asset_returns:      pd.DataFrame = None,
    asset_prices:       pd.DataFrame = None,
):
    app = dash.Dash(
        __name__,
        external_stylesheets=[dbc.themes.CYBORG, _FONTS],
        title="QuantCore — Viking Quant",
        suppress_callback_exceptions=True,
    )

    # ── Pre-generate AI portfolio insight ──────────────────────────
    ai_insight = portfolio_insight(tearsheet, risk_report, regime_stats)

    ts   = tearsheet["Value"]
    ts_str   = str(ts.to_dict())
    risk_str = str({k: str(v) for k, v in risk_report.items()})

    # ── KPI runestones ─────────────────────────────────────────────
    kpi_strip = dbc.Row([
        _kpi_stone("CAGR",         ts.get("CAGR",        "—"), GREEN,  "ᚠ"),
        _kpi_stone("Sharpe Ratio", ts.get("Sharpe Ratio", "—"), ICE,   "ᛊ"),
        _kpi_stone("Max Drawdown", ts.get("Max Drawdown", "—"), BLOOD, "ᛉ"),
        _kpi_stone("Final Equity", f"${backtest_result['equity'].iloc[-1]:,.0f}", GOLD, "ᛟ"),
    ], className="mb-3")

    # ══════════════════════════════════════════════════════════════
    # CHART BUILDERS — identical maths, Viking colour palette
    # ══════════════════════════════════════════════════════════════

    def fig_equity():
        fig = make_subplots(
            rows=3, cols=1, shared_xaxes=True,
            row_heights=[0.55, 0.25, 0.20], vertical_spacing=0.04,
            subplot_titles=["Portfolio Equity vs Benchmark",
                            "Drawdown %", "Daily Return %"],
        )
        fig.add_trace(go.Scatter(
            x=backtest_result.index, y=backtest_result["equity"],
            name="QuantCore", line=dict(color=GOLD, width=2.5),
            fill="tozeroy", fillcolor="rgba(201,162,39,0.06)",
        ), row=1, col=1)
        if "benchmark_equity" in backtest_result.columns:
            fig.add_trace(go.Scatter(
                x=backtest_result.index, y=backtest_result["benchmark_equity"],
                name="Benchmark", line=dict(color=STEEL, width=1.5, dash="dash"),
            ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=backtest_result.index, y=backtest_result["drawdown"] * 100,
            fill="tozeroy", fillcolor="rgba(139,0,0,0.25)",
            line=dict(color=BLOOD, width=1), name="Drawdown %",
        ), row=2, col=1)
        colors = [GREEN if r >= 0 else BLOOD for r in backtest_result["returns"]]
        fig.add_trace(go.Bar(
            x=backtest_result.index, y=backtest_result["returns"] * 100,
            marker_color=colors, name="Daily Return %",
        ), row=3, col=1)
        fig.update_layout(
            height=650, template="plotly_dark",
            paper_bgcolor=BG, plot_bgcolor=BG,
            font=dict(family=_FONT_B, color="white"),
            legend=dict(orientation="h", y=1.04),
            hovermode="x unified",
            margin=dict(l=40, r=20, t=40, b=20),
        )
        return fig

    def fig_3d_frontier():
        if efficient_frontier is None or efficient_frontier.empty:
            return go.Figure()
        ef  = efficient_frontier.dropna(subset=["return", "volatility", "sharpe"])
        vol = ef["volatility"].values
        ret = ef["return"].values
        sr  = ef["sharpe"].values
        from scipy.interpolate import griddata
        vol_lin = np.linspace(vol.min(), vol.max(), 40)
        ret_lin = np.linspace(ret.min(), ret.max(), 40)
        VOL, RET = np.meshgrid(vol_lin, ret_lin)
        SR = griddata((vol, ret), sr, (VOL, RET), method="cubic")
        fig = go.Figure()
        fig.add_trace(go.Surface(
            x=VOL, y=RET, z=SR,
            colorscale=[[0, "#3d0000"], [0.5, "#1a1040"], [1, "#c9a227"]],
            opacity=0.88,
            colorbar=dict(title="Sharpe", tickfont=dict(color="white")),
            contours=dict(z=dict(show=True, usecolormap=True, project_z=True)),
        ))
        fig.add_trace(go.Scatter3d(
            x=vol, y=ret, z=sr, mode="markers",
            marker=dict(size=4, color=sr,
                        colorscale=[[0,"#8b0000"],[1,"#c9a227"]]),
            name="Portfolios",
        ))
        idx_max = np.argmax(sr)
        fig.add_trace(go.Scatter3d(
            x=[vol[idx_max]], y=[ret[idx_max]], z=[sr[idx_max]],
            mode="markers+text",
            marker=dict(size=12, color=GOLD, symbol="diamond"),
            text=["⚔ Max Sharpe"], textfont=dict(color=GOLD, size=12),
            name="Optimal War Chest",
        ))
        fig.update_layout(
            title="ᛟ  3D Efficient Frontier — Return / Volatility / Sharpe",
            scene=dict(
                xaxis=dict(title="Volatility", backgroundcolor=BG, gridcolor=BORDER),
                yaxis=dict(title="Return",     backgroundcolor=BG, gridcolor=BORDER),
                zaxis=dict(title="Sharpe",     backgroundcolor=BG, gridcolor=BORDER),
                bgcolor=BG,
            ),
            template="plotly_dark", paper_bgcolor=BG,
            font=dict(family=_FONT_B, color="white"),
            height=540, margin=dict(l=0, r=0, t=50, b=0),
        )
        return fig

    def fig_3d_ic_surface():
        if ic_surface is None or ic_surface.empty:
            return go.Figure()
        surf     = ic_surface.dropna(how="all").fillna(0)
        dates    = surf.index
        horizons = surf.columns.astype(int).tolist()
        Z        = surf.values
        date_nums = np.arange(len(dates))
        H, D      = np.meshgrid(horizons, date_nums)
        fig = go.Figure(go.Surface(
            x=H, y=D, z=Z,
            colorscale=[[0,"#8b0000"], [0.5,"#1a1040"], [1,"#87ceeb"]],
            colorbar=dict(title="IC", tickfont=dict(color="white")),
            opacity=0.9,
            contours=dict(z=dict(show=True, usecolormap=True, project_z=True)),
        ))
        tick_step = max(1, len(dates) // 8)
        tick_vals = list(range(0, len(dates), tick_step))
        tick_text = [str(dates[i].date()) for i in tick_vals]
        fig.update_layout(
            title="ᚾ  3D Signal IC Surface — Predictive Power × Time × Horizon",
            scene=dict(
                xaxis=dict(title="Forward Horizon (days)", backgroundcolor=BG, gridcolor=BORDER),
                yaxis=dict(title="Date", tickvals=tick_vals, ticktext=tick_text,
                           backgroundcolor=BG, gridcolor=BORDER),
                zaxis=dict(title="IC", backgroundcolor=BG, gridcolor=BORDER),
                bgcolor=BG,
            ),
            template="plotly_dark", paper_bgcolor=BG,
            font=dict(family=_FONT_B, color="white"),
            height=520, margin=dict(l=0, r=0, t=50, b=0),
        )
        return fig

    def fig_3d_drawdown():
        if asset_prices is None:
            return go.Figure()
        prices = asset_prices.iloc[:, :8]
        wealth = (1 + np.log(prices / prices.shift(1)).fillna(0)).cumprod()
        dd_df  = (wealth / wealth.cummax() - 1) * 100
        step   = max(1, len(dd_df) // 80)
        dd_s   = dd_df.iloc[::step]
        assets    = list(dd_s.columns)
        dates     = dd_s.index
        date_nums = np.arange(len(dates))
        asset_nums = np.arange(len(assets))
        D, A = np.meshgrid(date_nums, asset_nums)
        Z    = dd_s.T.values
        fig  = go.Figure(go.Surface(
            x=D, y=A, z=Z,
            colorscale=[[0,"#8b0000"], [0.5,"#e65100"], [1,"#1a1040"]],
            colorbar=dict(title="Drawdown %", tickfont=dict(color="white")),
            opacity=0.88, reversescale=True,
        ))
        tick_step = max(1, len(dates) // 6)
        tick_vals = list(range(0, len(dates), tick_step))
        tick_text = [str(dates[i].date()) for i in tick_vals]
        fig.update_layout(
            title="ᛉ  3D Drawdown Surface — Asset × Time × Depth",
            scene=dict(
                xaxis=dict(title="Date", tickvals=tick_vals, ticktext=tick_text,
                           backgroundcolor=BG, gridcolor=BORDER),
                yaxis=dict(title="Asset", tickvals=list(asset_nums), ticktext=assets,
                           backgroundcolor=BG, gridcolor=BORDER),
                zaxis=dict(title="Drawdown %", backgroundcolor=BG, gridcolor=BORDER),
                bgcolor=BG,
            ),
            template="plotly_dark", paper_bgcolor=BG,
            font=dict(family=_FONT_B, color="white"),
            height=520, margin=dict(l=0, r=0, t=50, b=0),
        )
        return fig

    def fig_regime():
        if regime_probs is None:
            return go.Figure()
        fig = go.Figure()
        colors_map = {"bear": BLOOD, "sideways": FIRE, "bull": GREEN}
        for col in regime_probs.columns:
            fig.add_trace(go.Scatter(
                x=regime_probs.index, y=regime_probs[col],
                name=col.capitalize(),
                fill="tonexty" if col != regime_probs.columns[0] else "tozeroy",
                line=dict(color=colors_map.get(col, ICE), width=1),
                stackgroup="one",
            ))
        fig.update_layout(
            title="Market Fate — Regime Probabilities (Hidden Markov Model)",
            template="plotly_dark", paper_bgcolor=BG, plot_bgcolor=BG,
            font=dict(family=_FONT_B, color="white"),
            height=300, yaxis=dict(title="Probability", range=[0, 1]),
            legend=dict(orientation="h"), margin=dict(l=40, r=20, t=50, b=20),
        )
        return fig

    def fig_ic_decay():
        if ic_decay is None or ic_decay.empty:
            return go.Figure()
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=ic_decay.index, y=ic_decay["mean_ic"],
            name="Mean IC", line=dict(color=GOLD, width=2.5),
            fill="tozeroy", fillcolor="rgba(201,162,39,0.08)",
        ))
        fig.add_trace(go.Scatter(
            x=ic_decay.index, y=ic_decay["mean_ic"] + ic_decay["std_ic"],
            name="+1σ", line=dict(color=ICE, dash="dot", width=1),
        ))
        fig.add_trace(go.Scatter(
            x=ic_decay.index, y=ic_decay["mean_ic"] - ic_decay["std_ic"],
            name="-1σ", line=dict(color=ICE, dash="dot", width=1),
            fill="tonexty", fillcolor="rgba(135,206,235,0.06)",
        ))
        fig.add_hline(y=0.05, line_dash="dash", line_color=GREEN,
                      annotation_text="IC = 0.05  (meaningful)")
        fig.add_hline(y=0, line_dash="solid", line_color=STEEL)
        fig.update_layout(
            title="Elder Futhark Decay — Signal Predictive Power vs Forward Horizon",
            template="plotly_dark", paper_bgcolor=BG, plot_bgcolor=BG,
            font=dict(family=_FONT_B, color="white"),
            height=340, xaxis_title="Forward Horizon (days)", yaxis_title="IC",
            legend=dict(orientation="h"), margin=dict(l=40, r=20, t=50, b=20),
        )
        return fig

    def fig_var():
        labels = ["Hist 95%","Hist 99%","Normal","Student-t",
                  "Cornish-Fisher","Monte Carlo","CVaR 95%","CVaR 99%"]
        keys   = ["historical_var_95","historical_var_99","parametric_var_normal",
                  "parametric_var_t","cornish_fisher_var","monte_carlo_var","cvar_95","cvar_99"]
        values = [risk_report.get(k, 0) * 100 for k in keys]
        colors = [BLOOD if "cvar" in k else FIRE for k in keys]
        fig = go.Figure(go.Bar(
            x=labels, y=values, marker_color=colors,
            text=[f"{v:.2f}%" for v in values], textposition="outside",
        ))
        fig.update_layout(
            title="VaR & CVaR Estimates — Daily Battle Loss %",
            template="plotly_dark", paper_bgcolor=BG, plot_bgcolor=BG,
            font=dict(family=_FONT_B, color="white"),
            height=340, yaxis_title="Daily Loss %",
            margin=dict(l=40, r=20, t=50, b=40),
        )
        return fig

    def fig_distribution():
        r = backtest_result["returns"].dropna() * 100
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=r, nbinsx=80, name="Returns",
            marker_color=GOLD, opacity=0.60,
        ))
        mu, sigma = r.mean(), r.std()
        x_range = np.linspace(r.min(), r.max(), 200)
        y_norm  = norm.pdf(x_range, mu, sigma) * len(r) * (r.max() - r.min()) / 80
        fig.add_trace(go.Scatter(
            x=x_range, y=y_norm,
            name="Normal Fit", line=dict(color=ICE, width=2),
        ))
        fig.add_vline(x=float(np.percentile(r, 5)), line_dash="dash",
                      line_color=BLOOD, annotation_text="5th pct (VaR)")
        fig.update_layout(
            title="Return Distribution vs Normal — Fat Tails of Battle",
            template="plotly_dark", paper_bgcolor=BG, plot_bgcolor=BG,
            font=dict(family=_FONT_B, color="white"),
            height=330, xaxis_title="Daily Return %",
            margin=dict(l=40, r=20, t=50, b=20),
        )
        return fig

    def fig_rolling():
        fig = make_subplots(
            rows=2, cols=1, shared_xaxes=True,
            subplot_titles=["Rolling Sharpe Ratio (252 days)",
                            "Rolling Volatility % (63 days)"],
        )
        fig.add_trace(go.Scatter(
            x=rolling_sharpe.index, y=rolling_sharpe.values,
            name="Sharpe", line=dict(color=ICE, width=2),
        ), row=1, col=1)
        fig.add_hline(y=1, line_dash="dot", line_color=GREEN, row=1, col=1,
                      annotation_text="Sharpe = 1")
        fig.add_hline(y=0, line_dash="dash", line_color=STEEL, row=1, col=1)
        fig.add_trace(go.Scatter(
            x=rolling_vol.index, y=rolling_vol.values * 100,
            fill="tozeroy", fillcolor="rgba(135,206,235,0.10)",
            line=dict(color=ICE, width=2), name="Vol %",
        ), row=2, col=1)
        fig.update_layout(
            template="plotly_dark", paper_bgcolor=BG, plot_bgcolor=BG,
            font=dict(family=_FONT_B, color="white"),
            height=390, margin=dict(l=40, r=20, t=40, b=20),
        )
        return fig

    # ── Data table style ──────────────────────────────────────────
    tbl_style = dict(
        style_header={
            "backgroundColor": "#12102a",
            "color":           GOLD,
            "fontWeight":      "bold",
            "fontFamily":      _FONT_H,
            "letterSpacing":   "1px",
        },
        style_data={"backgroundColor": CARD, "color": "white", "fontFamily": _FONT_B},
        style_cell={"padding": "8px", "textAlign": "left",
                    "border": f"1px solid {BORDER}"},
        style_table={"overflowX": "auto"},
    )

    ts_records   = tearsheet.reset_index().rename(
        columns={"index": "Metric", 0: "Value"}).to_dict("records")
    risk_records = [{"Metric": k.replace("_", " ").title(), "Value": str(v)}
                    for k, v in risk_report.items()]
    regime_table = (
        regime_stats.reset_index().to_dict("records")
        if regime_stats is not None and not regime_stats.empty else []
    )

    # ══════════════════════════════════════════════════════════════
    # LAYOUT
    # ══════════════════════════════════════════════════════════════

    _base_text = {
        "fontFamily":   _FONT_B,
        "letterSpacing": "0.3px",
    }

    app.layout = dbc.Container(
        fluid=True,
        style={"backgroundColor": BG, "minHeight": "100vh", **_base_text},
        children=[

            # ── HEADER ─────────────────────────────────────────
            dbc.Row(dbc.Col(html.Div([

                # decorative rune strip
                html.Div(
                    "ᚠ ᚢ ᚦ ᚨ ᚱ ᚲ ᚷ ᚹ ᚺ ᚾ ᛁ ᛃ ᛇ ᛈ ᛉ ᛊ ᛏ ᛒ ᛖ ᛗ ᛚ ᛜ ᛞ ᛟ",
                    style={
                        "color":         GOLD,
                        "opacity":       "0.18",
                        "letterSpacing": "8px",
                        "fontSize":      "0.85rem",
                        "marginBottom":  "10px",
                        "fontFamily":    _FONT_H,
                        "overflow":      "hidden",
                        "whiteSpace":    "nowrap",
                    },
                ),

                dbc.Row([
                    dbc.Col([
                        html.A(
                            html.H1("QuantCore", style={
                                "color":       GOLD,
                                "fontWeight":  "900",
                                "fontSize":    "3rem",
                                "margin":      0,
                                "cursor":      "pointer",
                                "fontFamily":  _FONT_H,
                                "letterSpacing": "3px",
                                "textShadow":  f"0 0 30px {GOLD}55",
                            }),
                            href="/", style={"textDecoration": "none"},
                            title="Return to home",
                        ),
                        html.P(
                            "ᛟ  Institutional Quantitative Research  ·  Viking Precision  ·  "
                            "Hedge Fund Alpha",
                            style={
                                "color":         RUNE,
                                "margin":        "4px 0 0 2px",
                                "fontSize":      "0.88rem",
                                "fontFamily":    _FONT_H,
                                "letterSpacing": "1.5px",
                            },
                        ),
                    ], md=8),
                    dbc.Col(html.Div([
                        html.Span("⚔", style={"fontSize": "3.5rem", "opacity": "0.25",
                                              "marginRight": "6px"}),
                        html.Span("🛡", style={"fontSize": "3.5rem", "opacity": "0.25",
                                              "marginRight": "6px"}),
                        html.Span("⚔", style={"fontSize": "3.5rem", "opacity": "0.25"}),
                    ], style={"textAlign": "right", "paddingTop": "8px"}), md=4),
                ]),

                html.Hr(style={
                    "borderColor": GOLD,
                    "opacity":     "0.35",
                    "margin":      "12px 0",
                    "boxShadow":   f"0 0 12px {GOLD}44",
                }),

            ], style={"padding": "20px 0 8px 0", "marginBottom": "16px"}))),

            # ── VALHALLA METRICS ───────────────────────────────
            _rune_header("ᛟ", "VALHALLA METRICS", "Key performance runestones", GOLD),
            kpi_strip,

            # ── AI PORTFOLIO WAR COUNSEL ───────────────────────
            _rune_header("ᛗ", "ODIN'S WAR COUNSEL",
                         "Gemini AI — portfolio intelligence from the all-seeing oracle", ICE),
            _card([
                html.Div([
                    html.Span("🤖  Powered by Google Gemini AI", style={
                        "backgroundColor": "#0a2a4a",
                        "color":           ICE,
                        "fontSize":        "0.75rem",
                        "padding":         "3px 10px",
                        "borderRadius":    "20px",
                        "fontWeight":      "bold",
                        "marginBottom":    "12px",
                        "display":         "inline-block",
                        "letterSpacing":   "0.5px",
                    }),
                    html.Span(
                        " ✓ ACTIVE" if GEMINI_AVAILABLE else " ✗ OFFLINE — set GEMINI_API_KEY",
                        style={
                            "color":       GREEN if GEMINI_AVAILABLE else BLOOD,
                            "fontSize":    "0.72rem",
                            "marginLeft":  "8px",
                            "fontWeight":  "bold",
                        },
                    ),
                ]),

                # Auto-generated portfolio narrative
                html.Div([
                    html.P(ai_insight, style={
                        "color":        "#ddd",
                        "fontFamily":   _FONT_B,
                        "fontSize":     "0.95rem",
                        "lineHeight":   "1.7",
                        "fontStyle":    "italic",
                        "borderLeft":   f"3px solid {GOLD}",
                        "paddingLeft":  "16px",
                        "margin":       "12px 0 16px 0",
                    }),
                ]),

                # Ask Odin interactive chat
                html.H6("⚔  Ask Odin — Interactive AI Oracle", style={
                    "color":        GOLD,
                    "fontFamily":   _FONT_H,
                    "letterSpacing": "1px",
                    "marginBottom": "10px",
                    "marginTop":    "8px",
                }),
                html.P(
                    "Ask any question about the portfolio, strategy, risk, or market conditions.",
                    style={"color": MIST, "fontSize": "0.82rem", "marginBottom": "10px"},
                ),
                dbc.Row([
                    dbc.Col(dbc.Textarea(
                        id="odin-input",
                        placeholder="e.g. Why is the Sharpe ratio declining? "
                                    "How is the strategy performing in bear regimes? "
                                    "Should I reduce leverage?",
                        style={
                            "backgroundColor": CARD2,
                            "color":           "white",
                            "border":          f"1px solid {GOLD}55",
                            "borderRadius":    "8px",
                            "fontFamily":      _FONT_B,
                            "fontSize":        "0.9rem",
                            "padding":         "10px",
                            "minHeight":       "70px",
                        },
                        debounce=False,
                    ), width=10),
                    dbc.Col(dbc.Button(
                        ["ᚠ", html.Br(), "Consult"], id="odin-btn", n_clicks=0,
                        style={
                            "backgroundColor": GOLD,
                            "color":           "#000",
                            "fontWeight":      "bold",
                            "border":          "none",
                            "borderRadius":    "8px",
                            "width":           "100%",
                            "height":          "70px",
                            "fontSize":        "0.9rem",
                            "fontFamily":      _FONT_H,
                            "letterSpacing":   "1px",
                        },
                    ), width=2),
                ], className="mb-3"),
                dcc.Loading(
                    type="dot", color=GOLD,
                    children=html.Div(id="odin-output"),
                ),
            ], border_color=ICE),

            # ── SIGNAL FORGE ──────────────────────────────────
            _rune_header("ᛏ", "THE SIGNAL FORGE",
                         "Strategies forged from the world's greatest hedge fund research", FIRE),
            _card([
                html.P(
                    "QuantCore wields publicly documented quantitative strategies "
                    "wielded by elite funds — forged in peer-reviewed academic iron.",
                    style={"color": MIST, "marginBottom": "16px", "fontFamily": _FONT_B},
                ),
                dbc.Row([
                    dbc.Col(_card([
                        html.H6("⚔  Trend Following", style={"color": ICE, "fontFamily": _FONT_H,
                                                              "letterSpacing": "1px"}),
                        html.P("Man AHL · Winton · Campbell & Co — EWMA crossover + "
                               "time-series momentum (Moskowitz et al. 2012). Multi-speed "
                               "lookback windows combined for robustness. "
                               "Volatility-scaled so every position contributes equal risk.",
                               style={"color": "#bbb", "fontSize": "0.83rem",
                                      "fontFamily": _FONT_B}),
                    ], border_color=ICE), md=4),
                    dbc.Col(_card([
                        html.H6("🛡  Carry Factor", style={"color": GREEN, "fontFamily": _FONT_H,
                                                           "letterSpacing": "1px"}),
                        html.P("AQR Capital · Koijen, Moskowitz, Pedersen 2018. "
                               "Earnings yield proxy + roll-down carry + stability carry "
                               "blended cross-sectionally. Assets with higher carry "
                               "outperform across centuries of data.",
                               style={"color": "#bbb", "fontSize": "0.83rem",
                                      "fontFamily": _FONT_B}),
                    ], border_color=GREEN), md=4),
                    dbc.Col(_card([
                        html.H6("ᚠ  Multi-Factor Model", style={"color": RUNE, "fontFamily": _FONT_H,
                                                                 "letterSpacing": "1px"}),
                        html.P("AQR published factor research — Asness et al. "
                               "Value, Momentum, Quality, Low-Vol factors z-scored "
                               "cross-sectionally and blended. "
                               "IC/IR tracking validates predictive power continuously.",
                               style={"color": "#bbb", "fontSize": "0.83rem",
                                      "fontFamily": _FONT_B}),
                    ], border_color=RUNE), md=4),
                ]),
                dbc.Row([
                    dbc.Col(_card([
                        html.H6("ᛉ  Statistical Arbitrage", style={"color": FIRE, "fontFamily": _FONT_H,
                                                                    "letterSpacing": "1px"}),
                        html.P("Renaissance Technologies · D.E. Shaw style. "
                               "Engle-Granger cointegration identifies pairs. "
                               "Kalman Filter tracks dynamic hedge ratio. "
                               "Spread z-score drives mean-reversion entry/exit.",
                               style={"color": "#bbb", "fontSize": "0.83rem",
                                      "fontFamily": _FONT_B}),
                    ], border_color=FIRE), md=4),
                    dbc.Col(_card([
                        html.H6("ᛗ  ML Signal Combination", style={"color": GOLD, "fontFamily": _FONT_H,
                                                                     "letterSpacing": "1px"}),
                        html.P("Two Sigma · D.E. Shaw — Ridge, Elastic Net, Gradient "
                               "Boosting and Random Forest combine weak signals into "
                               "a stronger composite. Walk-forward validation prevents "
                               "look-ahead bias — the cardinal sin of quant ML.",
                               style={"color": "#bbb", "fontSize": "0.83rem",
                                      "fontFamily": _FONT_B}),
                    ], border_color=GOLD), md=4),
                    dbc.Col(_card([
                        html.H6("⚖  Risk Parity & HRP", style={"color": BLOOD, "fontFamily": _FONT_H,
                                                                 "letterSpacing": "1px"}),
                        html.P("Bridgewater All Weather · AQR. "
                               "Hierarchical Risk Parity (Lopez de Prado 2016) and "
                               "Equal Risk Contribution ensure no asset dominates. "
                               "Robust to correlation breakdown during crises.",
                               style={"color": "#bbb", "fontSize": "0.83rem",
                                      "fontFamily": _FONT_B}),
                    ], border_color=BLOOD), md=4),
                ]),
            ], border_color=FIRE),

            # ── STOCK ORACLE ───────────────────────────────────
            _rune_header("🔭", "ORACLE'S VISION",
                         "Real-time stock divination — financials · news · AI analysis", GOLD),
            _card([
                html.P(
                    "Enter any publicly traded ticker to receive real-time financials, "
                    "a multi-factor quant verdict, Gemini AI analyst commentary, "
                    "and the latest breaking news from Yahoo Finance and the New York Times.",
                    style={"color": MIST, "marginBottom": "12px", "fontFamily": _FONT_B},
                ),
                dbc.Row([
                    dbc.Col(dbc.Input(
                        id="stock-input",
                        placeholder="Enter ticker  (e.g.  AAPL · NVDA · TSLA · MSFT)...",
                        type="text",
                        style={
                            "backgroundColor": CARD2,
                            "color":           "white",
                            "border":          f"1px solid {GOLD}",
                            "borderRadius":    "8px",
                            "fontSize":        "1rem",
                            "padding":         "11px",
                            "fontFamily":      _FONT_B,
                        },
                    ), width=9),
                    dbc.Col(dbc.Button(
                        "⚔  Analyse", id="stock-btn", n_clicks=0,
                        style={
                            "backgroundColor": GOLD,
                            "color":           "#000",
                            "fontWeight":      "bold",
                            "border":          "none",
                            "borderRadius":    "8px",
                            "width":           "100%",
                            "fontSize":        "1rem",
                            "fontFamily":      _FONT_H,
                            "letterSpacing":   "1px",
                        },
                    ), width=3),
                ], className="mb-3"),
                dcc.Loading(type="circle", color=GOLD,
                            children=html.Div(id="stock-output")),
            ]),

            # ── BATTLE RECORD ─────────────────────────────────
            _rune_header("ᛊ", "THE BATTLE RECORD",
                         "Portfolio performance — equity · drawdown · returns", ICE),
            _card([
                dcc.Graph(figure=fig_equity(), config={"displayModeBar": False}),
                _desc("Top: portfolio equity (gold) vs benchmark (steel dashed). "
                      "Middle: drawdown — how far below peak at each point. "
                      "Bottom: daily return bars — green = gain, red = loss."),
            ], border_color=ICE),

            dbc.Row([
                dbc.Col(_card([
                    dcc.Graph(figure=fig_rolling(), config={"displayModeBar": False}),
                    _desc("Rolling 252-day Sharpe (top) — above 1.0 (green) is excellent. "
                          "Rolling 63-day volatility (bottom) shows risk through time."),
                ], border_color=ICE), md=7),
                dbc.Col(_card([
                    dcc.Graph(figure=fig_var(), config={"displayModeBar": False}),
                    _desc("VaR = max expected daily loss at 95%/99% confidence. "
                          "CVaR (blood red) = average loss in worst-case scenarios. "
                          "Multiple methods: historical, parametric, Cornish-Fisher, Monte Carlo."),
                ], border_color=BLOOD), md=5),
            ]),

            _card([
                dcc.Graph(figure=fig_distribution(), config={"displayModeBar": False}),
                _desc("Daily returns histogram vs fitted normal (ice curve). "
                      "Fat tails = more extreme days than normal predicts (excess kurtosis). "
                      "Red dashed line = 5th percentile (historical VaR)."),
            ], border_color=RUNE),

            # ── THE NORNS — REGIME ─────────────────────────────
            _rune_header("ᚾ", "THE NORNS — FATE OF MARKETS",
                         "Hidden Markov Model regime classification — bull · sideways · bear", GREEN),
            _card([
                dcc.Graph(figure=fig_regime(), config={"displayModeBar": False}),
                _desc("The Norns (Norse fates) weave three market states: "
                      "Bull (green), Sideways (ember), Bear (blood). "
                      "Stacked area = probability of each regime over time from a 3-state HMM."),
            ], border_color=GREEN),
            _card([
                html.H5("Regime Battle Statistics", style={
                    "color": GREEN, "marginBottom": "10px", "fontFamily": _FONT_H,
                }),
                dash_table.DataTable(
                    data=regime_table,
                    columns=[{"name": c, "id": c} for c in (
                        regime_stats.reset_index().columns
                        if regime_stats is not None and not regime_stats.empty else []
                    )],
                    **tbl_style,
                ) if regime_table else html.P("No regime data.", style={"color": MIST}),
                _desc("Mean return, volatility, and Sharpe by regime. "
                      "High Bull Sharpe + negative Bear Sharpe confirms the HMM is battle-ready."),
            ], border_color=GREEN),

            # ── ELDER FUTHARK — SIGNAL DECAY ──────────────────
            _rune_header("ᚠ", "ELDER FUTHARK — SIGNAL DECAY",
                         "Information Coefficient decay — how long the signal holds its edge", RUNE),
            _card([
                dcc.Graph(figure=fig_ic_decay(), config={"displayModeBar": False}),
                _desc("IC > 0.05 (green line) is economically significant. "
                      "Fast decay = short-term signal; slow decay = works over months. "
                      "Shaded band = ±1 standard deviation of IC across dates."),
            ], border_color=RUNE),

            # ── WAR MAPS — 3D ─────────────────────────────────
            _rune_header("ᛟ", "THE WAR MAPS — 3D BATTLE CHARTS",
                         "Drag to rotate · scroll to zoom · Norse dimensions", GOLD),

            _card([
                dcc.Graph(figure=fig_3d_frontier(), config={"scrollZoom": True}),
                _desc("3D efficient frontier surface. X = risk, Y = return, Z = Sharpe. "
                      "The gold diamond marks the max Sharpe portfolio — the optimal war chest. "
                      "Gold peaks = highest risk-adjusted return in that region of risk/return space."),
            ]),

            _card([
                dcc.Graph(figure=fig_3d_ic_surface(), config={"scrollZoom": True}),
                _desc("3D map of signal predictive power (IC) across time (Y) and forward horizon (X). "
                      "Ice peaks = signal was highly predictive. "
                      "Blood troughs = signal lost its edge (market dislocations)."),
            ]),

            _card([
                dcc.Graph(figure=fig_3d_drawdown(), config={"scrollZoom": True}),
                _desc("3D drawdown surface: each asset (Y) over time (X) at each drawdown depth (Z). "
                      "Deep blood valleys = severe crashes in that asset at that moment. "
                      "Use this to identify which warrior dragged the war party and when."),
            ]),

            # ── THE SAGA — FULL RECORD ─────────────────────────
            _rune_header("📜", "THE SAGA — BATTLE CHRONICLES",
                         "Complete performance tearsheet and risk report", FIRE),
            dbc.Row([
                dbc.Col(_card([
                    html.H5("Performance Tearsheet", style={
                        "color": ICE, "marginBottom": "10px", "fontFamily": _FONT_H,
                    }),
                    dash_table.DataTable(
                        data=ts_records,
                        columns=[{"name": c, "id": c} for c in ["Metric", "Value"]],
                        **tbl_style,
                    ),
                    _desc("Return, Sharpe, Sortino, Calmar, Alpha, Beta, IR vs benchmark."),
                ], border_color=ICE), md=6),
                dbc.Col(_card([
                    html.H5("Risk Report", style={
                        "color": BLOOD, "marginBottom": "10px", "fontFamily": _FONT_H,
                    }),
                    dash_table.DataTable(
                        data=risk_records,
                        columns=[{"name": "Metric", "id": "Metric"},
                                 {"name": "Value",  "id": "Value"}],
                        **tbl_style,
                    ),
                    _desc("VaR · CVaR · Max Drawdown · Skewness · Kurtosis — the full risk saga."),
                ], border_color=BLOOD), md=6),
            ], className="mb-5"),

            # ── FOOTER ────────────────────────────────────────
            html.Hr(style={"borderColor": GOLD, "opacity": "0.2"}),
            html.Div(
                "QuantCore  ·  ᚠ ᚢ ᚦ ᚨ ᚱ ᚲ  ·  Institutional Quant Research  ·  "
                "All strategies derived from publicly available academic literature.",
                style={
                    "color":         RUNE,
                    "fontSize":      "0.75rem",
                    "textAlign":     "center",
                    "padding":       "16px 0 32px 0",
                    "fontFamily":    _FONT_H,
                    "letterSpacing": "1px",
                },
            ),
        ],
    )

    # ══════════════════════════════════════════════════════════════
    # CALLBACKS
    # ══════════════════════════════════════════════════════════════

    @app.callback(
        Output("odin-output", "children"),
        Input("odin-btn",  "n_clicks"),
        State("odin-input", "value"),
        prevent_initial_call=True,
    )
    def consult_odin(n_clicks, question):
        if not question or not question.strip():
            return html.P("Speak your question, warrior.",
                          style={"color": BLOOD, "fontFamily": _FONT_B})

        answer = ask_odin(question.strip(), ts_str, risk_str)

        return html.Div([
            html.Div("ᛗ  Odin Speaks:", style={
                "color":       GOLD,
                "fontFamily":  _FONT_H,
                "fontSize":    "0.9rem",
                "fontWeight":  "bold",
                "marginBottom": "6px",
                "letterSpacing": "1px",
            }),
            html.P(answer, style={
                "color":        "#ddd",
                "fontFamily":   _FONT_B,
                "fontSize":     "0.95rem",
                "lineHeight":   "1.7",
                "borderLeft":   f"3px solid {ICE}",
                "paddingLeft":  "14px",
                "margin":       0,
                "whiteSpace":   "pre-wrap",
            }),
        ], style={
            "backgroundColor": "#0a0a1e",
            "border":          f"1px solid {ICE}33",
            "borderRadius":    "8px",
            "padding":         "16px",
        })

    @app.callback(
        Output("stock-output", "children"),
        Input("stock-btn",  "n_clicks"),
        State("stock-input", "value"),
        prevent_initial_call=True,
    )
    def search_stock(n_clicks, ticker):
        if not ticker or not ticker.strip():
            return html.P("Enter a ticker symbol, warrior.",
                          style={"color": BLOOD})

        data = get_stock_data(ticker.strip().upper())

        if data["error"]:
            return html.P(f"Oracle error: {data['error']}",
                          style={"color": BLOOD})

        metrics   = data["metrics"]
        valuation = data["valuation"]
        news      = data["news"]
        history   = data["price_history"]

        # Price chart (Vikings palette)
        price_fig = go.Figure()
        if not history.empty:
            price_fig.add_trace(go.Scatter(
                x=history.index, y=history.values,
                name="Price", line=dict(color=GOLD, width=2.5),
                fill="tozeroy", fillcolor="rgba(201,162,39,0.07)",
            ))
            if len(history) >= 50:
                ma50 = history.rolling(50).mean()
                price_fig.add_trace(go.Scatter(
                    x=ma50.index, y=ma50.values,
                    name="50D MA", line=dict(color=ICE, width=1.5, dash="dot"),
                ))
            if len(history) >= 200:
                ma200 = history.rolling(200).mean()
                price_fig.add_trace(go.Scatter(
                    x=ma200.index, y=ma200.values,
                    name="200D MA", line=dict(color=STEEL, width=1.5, dash="dash"),
                ))
        price_fig.update_layout(
            title=f"{metrics.get('Name', ticker)} — 1 Year Price History",
            template="plotly_dark", paper_bgcolor=CARD, plot_bgcolor=CARD,
            font=dict(family=_FONT_B, color="white"),
            height=330, yaxis_title="Price ($)",
            legend=dict(orientation="h"),
            margin=dict(l=40, r=20, t=50, b=20),
        )

        # Verdict badge
        v_color  = valuation.get("color", "white")
        v_text   = valuation.get("verdict", "N/A")
        verdict_box = html.Div([
            html.H2(v_text, style={
                "color":      v_color,
                "fontWeight": "900",
                "fontSize":   "2rem",
                "fontFamily": _FONT_H,
                "margin":     "0 0 8px 0",
                "letterSpacing": "2px",
            }),
            html.P(f"Composite score: {valuation.get('score', 0):+d}",
                   style={"color": MIST, "margin": 0, "fontFamily": _FONT_B}),
        ], style={
            "textAlign":    "center",
            "padding":      "20px",
            "border":       f"2px solid {v_color}",
            "borderRadius": "10px",
            "marginBottom": "12px",
            "boxShadow":    f"0 0 20px {v_color}33",
        })

        reasons = html.Ul([
            html.Li(r, style={"color": "#ccc", "marginBottom": "4px",
                              "fontFamily": _FONT_B})
            for r in valuation.get("reasons", [])
        ], style={"paddingLeft": "16px"})

        # Gemini AI commentary
        ai_note = stock_ai_commentary(ticker.strip().upper(), metrics, valuation)
        ai_block = html.Div([
            html.Div("ᛗ  Gemini AI Analyst Note:", style={
                "color":       GOLD,
                "fontFamily":  _FONT_H,
                "fontSize":    "0.85rem",
                "fontWeight":  "bold",
                "marginBottom": "6px",
                "letterSpacing": "1px",
            }),
            html.P(ai_note, style={
                "color":       "#ddd",
                "fontFamily":  _FONT_B,
                "fontSize":    "0.9rem",
                "lineHeight":  "1.65",
                "borderLeft":  f"3px solid {GOLD}",
                "paddingLeft": "12px",
                "margin":      0,
            }),
        ], style={
            "backgroundColor": "#0c0c00",
            "border":          f"1px solid {GOLD}33",
            "borderRadius":    "8px",
            "padding":         "12px",
            "marginBottom":    "12px",
        }) if ai_note else None

        # Metrics table
        metrics_rows = [{"Metric": k, "Value": v} for k, v in metrics.items()]
        metrics_tbl  = dash_table.DataTable(
            data=metrics_rows,
            columns=[{"name": "Metric", "id": "Metric"}, {"name": "Value", "id": "Value"}],
            **tbl_style,
        )

        # News cards
        news_items = []
        for article in news[:8]:
            badge_color = ICE if article["source"] == "Yahoo Finance" else "#aa0000"
            news_items.append(html.Div([
                html.Div([
                    html.Span(article["source"], style={
                        "backgroundColor": badge_color,
                        "color":           "#000" if article["source"] == "Yahoo Finance" else "white",
                        "fontSize":        "0.7rem",
                        "padding":         "2px 8px",
                        "borderRadius":    "4px",
                        "marginRight":     "8px",
                        "fontWeight":      "bold",
                        "fontFamily":      _FONT_B,
                    }),
                    html.Span(article.get("date", ""), style={
                        "color": MIST, "fontSize": "0.75rem",
                    }),
                ], style={"marginBottom": "4px"}),
                html.A(article["title"],
                       href=article.get("url", "#"), target="_blank",
                       style={
                           "color":          GOLD,
                           "fontWeight":     "600",
                           "textDecoration": "none",
                           "fontSize":       "0.93rem",
                           "fontFamily":     _FONT_B,
                       }),
                html.P(article.get("summary", ""), style={
                    "color":    MIST,
                    "fontSize": "0.8rem",
                    "margin":   "4px 0 0 0",
                    "fontFamily": _FONT_B,
                }),
            ], style={
                "padding":         "12px",
                "marginBottom":    "8px",
                "backgroundColor": CARD2,
                "borderRadius":    "8px",
                "borderLeft":      f"3px solid {badge_color}",
            }))

        return html.Div([
            dbc.Row([
                dbc.Col([
                    verdict_box,
                    html.H6("⚔  Signal Drivers:", style={"color": MIST,
                                                          "fontFamily": _FONT_H}),
                    reasons,
                    ai_block,
                ], md=4),
                dbc.Col(dcc.Graph(figure=price_fig,
                                  config={"displayModeBar": False}), md=8),
            ], className="mb-3"),
            dbc.Row([
                dbc.Col([
                    html.H5("Key Financials", style={"color": ICE, "marginBottom": "8px",
                                                     "fontFamily": _FONT_H}),
                    metrics_tbl,
                ], md=5),
                dbc.Col([
                    html.H5("Breaking Intelligence", style={"color": GOLD, "marginBottom": "8px",
                                                             "fontFamily": _FONT_H}),
                    html.Div(news_items if news_items
                             else html.P("No news found.", style={"color": MIST})),
                ], md=7),
            ]),
        ])

    return app
