"""
QuantCore Interactive Dashboard — with 3D visualizations and regime analysis.

Run via: python examples/run_strategy.py
Then open: http://127.0.0.1:8050
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy.stats import norm

import dash
from dash import dcc, html, dash_table
import dash_bootstrap_components as dbc


# ── colour palette ─────────────────────────────────────────────────
GOLD   = "#f5c518"
CYAN   = "#00d4ff"
RED    = "#ef5350"
GREEN  = "#26a69a"
PURPLE = "#7e57c2"
ORANGE = "#ff7043"
BG     = "#0d0d1a"
CARD   = "#13132a"
BORDER = "#1e1e3a"


def _card(children, **kwargs):
    return dbc.Card(
        dbc.CardBody(children),
        style={"background": CARD, "border": f"1px solid {BORDER}",
               "borderRadius": "12px", "marginBottom": "16px"},
        **kwargs,
    )


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
        external_stylesheets=[dbc.themes.CYBORG],
        title="QuantCore",
    )

    # ── KPI strip ──────────────────────────────────────────────────
    def kpi(label, value, color=CYAN):
        return dbc.Col(_card([
            html.P(label, style={"color": "#aaa", "fontSize": "0.75rem",
                                 "marginBottom": "2px", "textTransform": "uppercase"}),
            html.H4(value, style={"color": color, "fontWeight": "bold", "margin": 0}),
        ]), xs=6, md=3)

    ts = tearsheet["Value"]
    kpis = dbc.Row([
        kpi("CAGR",           ts.get("CAGR",         "—"), GREEN),
        kpi("Sharpe",         ts.get("Sharpe Ratio",  "—"), CYAN),
        kpi("Max Drawdown",   ts.get("Max Drawdown",  "—"), RED),
        kpi("Final Equity",   f"${backtest_result['equity'].iloc[-1]:,.0f}", GOLD),
    ], className="mb-3")

    # ══════════════════════════════════════════════════════════════
    # FIG 1 — Equity + Drawdown + Returns (2D)
    # ══════════════════════════════════════════════════════════════
    def fig_equity():
        fig = make_subplots(
            rows=3, cols=1, shared_xaxes=True,
            row_heights=[0.55, 0.25, 0.20],
            vertical_spacing=0.04,
            subplot_titles=["Portfolio Equity", "Drawdown %", "Daily Returns %"],
        )
        fig.add_trace(go.Scatter(
            x=backtest_result.index, y=backtest_result["equity"],
            name="Portfolio", line=dict(color=CYAN, width=2),
            fill="tozeroy", fillcolor="rgba(0,212,255,0.05)",
        ), row=1, col=1)

        if "benchmark_equity" in backtest_result.columns:
            fig.add_trace(go.Scatter(
                x=backtest_result.index, y=backtest_result["benchmark_equity"],
                name="Benchmark", line=dict(color=ORANGE, width=1.5, dash="dash"),
            ), row=1, col=1)

        # Colour equity by regime
        if regime_labels is not None:
            regime_colors = {0: "rgba(239,83,80,0.15)", 1: "rgba(255,255,255,0.05)", 2: "rgba(38,166,154,0.12)"}
            rl = regime_labels.reindex(backtest_result.index).ffill()
            for state, col in regime_colors.items():
                mask = rl == state
                if not mask.any():
                    continue
                fig.add_trace(go.Scatter(
                    x=backtest_result.index[mask],
                    y=backtest_result["equity"][mask],
                    mode="markers", marker=dict(size=2, color=col),
                    showlegend=False, hoverinfo="skip",
                ), row=1, col=1)

        fig.add_trace(go.Scatter(
            x=backtest_result.index, y=backtest_result["drawdown"] * 100,
            fill="tozeroy", fillcolor="rgba(239,83,80,0.2)",
            line=dict(color=RED, width=1), name="Drawdown %",
        ), row=2, col=1)

        colors = [GREEN if r >= 0 else RED for r in backtest_result["returns"]]
        fig.add_trace(go.Bar(
            x=backtest_result.index, y=backtest_result["returns"] * 100,
            marker_color=colors, name="Daily Return %",
        ), row=3, col=1)

        fig.update_layout(height=650, template="plotly_dark",
                          paper_bgcolor=BG, plot_bgcolor=BG,
                          legend=dict(orientation="h", y=1.04),
                          hovermode="x unified", margin=dict(l=40, r=20, t=40, b=20))
        return fig

    # ══════════════════════════════════════════════════════════════
    # FIG 2 — 3D Efficient Frontier Surface
    # ══════════════════════════════════════════════════════════════
    def fig_3d_frontier():
        if efficient_frontier is None or efficient_frontier.empty:
            return go.Figure()

        ef = efficient_frontier.dropna(subset=["return", "volatility", "sharpe"])
        vol = ef["volatility"].values
        ret = ef["return"].values
        sr  = ef["sharpe"].values

        # Create a meshgrid surface by interpolating sharpe over vol/ret grid
        from scipy.interpolate import griddata
        vol_lin = np.linspace(vol.min(), vol.max(), 40)
        ret_lin = np.linspace(ret.min(), ret.max(), 40)
        VOL, RET = np.meshgrid(vol_lin, ret_lin)
        SR = griddata((vol, ret), sr, (VOL, RET), method="cubic")

        fig = go.Figure()
        fig.add_trace(go.Surface(
            x=VOL, y=RET, z=SR,
            colorscale="Viridis",
            colorbar=dict(title="Sharpe", tickfont=dict(color="white")),
            opacity=0.85,
            contours=dict(z=dict(show=True, usecolormap=True, highlightcolor="white", project_z=True)),
        ))
        fig.add_trace(go.Scatter3d(
            x=vol, y=ret, z=sr,
            mode="markers",
            marker=dict(size=4, color=sr, colorscale="Viridis",
                        line=dict(color="white", width=0.5)),
            name="Portfolios",
        ))

        # Mark max Sharpe point
        idx_max = np.argmax(sr)
        fig.add_trace(go.Scatter3d(
            x=[vol[idx_max]], y=[ret[idx_max]], z=[sr[idx_max]],
            mode="markers+text",
            marker=dict(size=10, color=GOLD, symbol="diamond"),
            text=["Max Sharpe"], textfont=dict(color=GOLD),
            name="Max Sharpe",
        ))

        fig.update_layout(
            title="3D Efficient Frontier — Return / Volatility / Sharpe",
            scene=dict(
                xaxis=dict(title="Volatility", backgroundcolor=BG, gridcolor=BORDER),
                yaxis=dict(title="Return",     backgroundcolor=BG, gridcolor=BORDER),
                zaxis=dict(title="Sharpe",     backgroundcolor=BG, gridcolor=BORDER),
                bgcolor=BG,
            ),
            template="plotly_dark", paper_bgcolor=BG,
            height=550, margin=dict(l=0, r=0, t=50, b=0),
        )
        return fig

    # ══════════════════════════════════════════════════════════════
    # FIG 3 — 3D Rolling IC Surface (signal decay over time × horizon)
    # ══════════════════════════════════════════════════════════════
    def fig_3d_ic_surface():
        if ic_surface is None or ic_surface.empty:
            return go.Figure()

        surf = ic_surface.dropna(how="all").fillna(0)
        dates    = surf.index
        horizons = surf.columns.astype(int).tolist()
        Z        = surf.values

        date_nums = np.arange(len(dates))
        H, D      = np.meshgrid(horizons, date_nums)

        fig = go.Figure(go.Surface(
            x=H, y=D, z=Z,
            colorscale=[
                [0.0,  "rgb(239,83,80)"],
                [0.5,  "rgb(30,30,60)"],
                [1.0,  "rgb(38,166,154)"],
            ],
            colorbar=dict(
                title="IC", tickfont=dict(color="white"),
                tickvals=[-0.1, 0, 0.1],
            ),
            opacity=0.9,
            contours=dict(z=dict(show=True, usecolormap=True, project_z=True)),
        ))

        # Date tick labels
        tick_step = max(1, len(dates) // 8)
        tick_vals = list(range(0, len(dates), tick_step))
        tick_text = [str(dates[i].date()) for i in tick_vals]

        fig.update_layout(
            title="3D Signal IC Surface — Predictive Power over Time & Horizon",
            scene=dict(
                xaxis=dict(title="Forward Horizon (days)", backgroundcolor=BG, gridcolor=BORDER),
                yaxis=dict(title="Date", tickvals=tick_vals, ticktext=tick_text,
                           backgroundcolor=BG, gridcolor=BORDER),
                zaxis=dict(title="Information Coefficient", backgroundcolor=BG, gridcolor=BORDER),
                bgcolor=BG,
            ),
            template="plotly_dark", paper_bgcolor=BG,
            height=520, margin=dict(l=0, r=0, t=50, b=0),
        )
        return fig

    # ══════════════════════════════════════════════════════════════
    # FIG 4 — 3D Drawdown Surface (asset × time × depth)
    # ══════════════════════════════════════════════════════════════
    def fig_3d_drawdown():
        if asset_prices is None:
            return go.Figure()

        prices = asset_prices.iloc[:, :8]   # top 8 assets for clarity
        wealth = (1 + np.log(prices / prices.shift(1)).fillna(0)).cumprod()
        dd_df  = (wealth / wealth.cummax() - 1) * 100   # percentage

        # Downsample dates for 3D performance
        step  = max(1, len(dd_df) // 80)
        dd_s  = dd_df.iloc[::step]

        assets   = list(dd_s.columns)
        dates    = dd_s.index
        date_nums = np.arange(len(dates))
        asset_nums = np.arange(len(assets))
        D, A      = np.meshgrid(date_nums, asset_nums)
        Z         = dd_s.T.values

        fig = go.Figure(go.Surface(
            x=D, y=A, z=Z,
            colorscale=[
                [0.0, "rgb(239,83,80)"],
                [0.5, "rgb(255,152,0)"],
                [1.0, "rgb(30,30,60)"],
            ],
            colorbar=dict(title="Drawdown %", tickfont=dict(color="white")),
            opacity=0.88,
            reversescale=True,
        ))

        tick_step = max(1, len(dates) // 6)
        tick_vals = list(range(0, len(dates), tick_step))
        tick_text = [str(dates[i].date()) for i in tick_vals]

        fig.update_layout(
            title="3D Drawdown Surface — Asset × Time × Depth",
            scene=dict(
                xaxis=dict(title="Date", tickvals=tick_vals, ticktext=tick_text,
                           backgroundcolor=BG, gridcolor=BORDER),
                yaxis=dict(title="Asset", tickvals=list(asset_nums),
                           ticktext=assets, backgroundcolor=BG, gridcolor=BORDER),
                zaxis=dict(title="Drawdown %", backgroundcolor=BG, gridcolor=BORDER),
                bgcolor=BG,
            ),
            template="plotly_dark", paper_bgcolor=BG,
            height=520, margin=dict(l=0, r=0, t=50, b=0),
        )
        return fig

    # ══════════════════════════════════════════════════════════════
    # FIG 5 — Regime probability area chart
    # ══════════════════════════════════════════════════════════════
    def fig_regime():
        if regime_probs is None:
            return go.Figure()

        fig = go.Figure()
        colors_map = {"bear": RED, "sideways": ORANGE, "bull": GREEN}
        for col in regime_probs.columns:
            fig.add_trace(go.Scatter(
                x=regime_probs.index, y=regime_probs[col],
                name=col.capitalize(),
                fill="tonexty" if col != regime_probs.columns[0] else "tozeroy",
                line=dict(color=colors_map.get(col, CYAN), width=1),
                stackgroup="one",
            ))

        fig.update_layout(
            title="Market Regime Probabilities (HMM)",
            template="plotly_dark", paper_bgcolor=BG, plot_bgcolor=BG,
            height=300, yaxis=dict(title="Probability", range=[0, 1]),
            legend=dict(orientation="h"),
            margin=dict(l=40, r=20, t=50, b=20),
        )
        return fig

    # ══════════════════════════════════════════════════════════════
    # FIG 6 — IC Decay curve + half-life marker
    # ══════════════════════════════════════════════════════════════
    def fig_ic_decay():
        if ic_decay is None or ic_decay.empty:
            return go.Figure()

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=ic_decay.index, y=ic_decay["mean_ic"],
            name="Mean IC", line=dict(color=CYAN, width=2),
            fill="tozeroy", fillcolor="rgba(0,212,255,0.08)",
        ))
        fig.add_trace(go.Scatter(
            x=ic_decay.index,
            y=ic_decay["mean_ic"] + ic_decay["std_ic"],
            name="+1σ", line=dict(color=PURPLE, dash="dot", width=1),
        ))
        fig.add_trace(go.Scatter(
            x=ic_decay.index,
            y=ic_decay["mean_ic"] - ic_decay["std_ic"],
            name="-1σ", line=dict(color=PURPLE, dash="dot", width=1),
            fill="tonexty", fillcolor="rgba(126,87,194,0.08)",
        ))
        fig.add_hline(y=0.05, line_dash="dash", line_color=GREEN,
                      annotation_text="IC=0.05 (meaningful)")
        fig.add_hline(y=0, line_dash="solid", line_color="grey")

        fig.update_layout(
            title="Factor IC Decay — Signal Predictive Power vs Forward Horizon",
            template="plotly_dark", paper_bgcolor=BG, plot_bgcolor=BG,
            height=320, xaxis_title="Forward Horizon (days)",
            yaxis_title="Information Coefficient",
            legend=dict(orientation="h"),
            margin=dict(l=40, r=20, t=50, b=20),
        )
        return fig

    # ══════════════════════════════════════════════════════════════
    # FIG 7 — VaR bar chart
    # ══════════════════════════════════════════════════════════════
    def fig_var():
        labels = ["Hist 95%", "Hist 99%", "Normal", "Student-t",
                  "Cornish-Fisher", "Monte Carlo", "CVaR 95%", "CVaR 99%"]
        keys   = ["historical_var_95", "historical_var_99", "parametric_var_normal",
                  "parametric_var_t", "cornish_fisher_var", "monte_carlo_var",
                  "cvar_95", "cvar_99"]
        values = [risk_report.get(k, 0) * 100 for k in keys]
        colors = [RED if "cvar" in k else ORANGE for k in keys]

        fig = go.Figure(go.Bar(
            x=labels, y=values, marker_color=colors,
            text=[f"{v:.2f}%" for v in values], textposition="outside",
        ))
        fig.update_layout(
            title="VaR & CVaR Estimates (Daily Loss %)",
            template="plotly_dark", paper_bgcolor=BG, plot_bgcolor=BG,
            height=330, yaxis_title="Daily Loss %",
            margin=dict(l=40, r=20, t=50, b=40),
        )
        return fig

    # ══════════════════════════════════════════════════════════════
    # FIG 8 — Return distribution
    # ══════════════════════════════════════════════════════════════
    def fig_distribution():
        r = backtest_result["returns"].dropna() * 100
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=r, nbinsx=80, name="Returns",
            marker_color=CYAN, opacity=0.65,
        ))
        mu, sigma = r.mean(), r.std()
        x_range = np.linspace(r.min(), r.max(), 200)
        y_norm  = norm.pdf(x_range, mu, sigma) * len(r) * (r.max() - r.min()) / 80
        fig.add_trace(go.Scatter(
            x=x_range, y=y_norm,
            name="Normal Fit", line=dict(color=ORANGE, width=2),
        ))
        fig.add_vline(x=float(np.percentile(r, 5)),
                      line_dash="dash", line_color=RED,
                      annotation_text="5th pct (VaR)")
        fig.update_layout(
            title="Return Distribution vs Normal",
            template="plotly_dark", paper_bgcolor=BG, plot_bgcolor=BG,
            height=320, xaxis_title="Daily Return %",
            margin=dict(l=40, r=20, t=50, b=20),
        )
        return fig

    # ══════════════════════════════════════════════════════════════
    # FIG 9 — Rolling Sharpe + Vol
    # ══════════════════════════════════════════════════════════════
    def fig_rolling():
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                            subplot_titles=["Rolling Sharpe (252d)", "Rolling Volatility % (63d)"])
        fig.add_trace(go.Scatter(
            x=rolling_sharpe.index, y=rolling_sharpe.values,
            name="Sharpe", line=dict(color=PURPLE, width=2),
        ), row=1, col=1)
        fig.add_hline(y=1, line_dash="dot", line_color=GREEN, row=1, col=1,
                      annotation_text="Sharpe=1")
        fig.add_hline(y=0, line_dash="dash", line_color="grey", row=1, col=1)

        fig.add_trace(go.Scatter(
            x=rolling_vol.index, y=rolling_vol.values * 100,
            fill="tozeroy", fillcolor="rgba(126,87,194,0.12)",
            line=dict(color=PURPLE, width=2), name="Vol %",
        ), row=2, col=1)

        fig.update_layout(
            template="plotly_dark", paper_bgcolor=BG, plot_bgcolor=BG,
            height=380, margin=dict(l=40, r=20, t=40, b=20),
        )
        return fig

    # ══════════════════════════════════════════════════════════════
    # Tables
    # ══════════════════════════════════════════════════════════════
    tbl_style = dict(
        style_header={"backgroundColor": "#1a1a3a", "color": GOLD, "fontWeight": "bold"},
        style_data={"backgroundColor": CARD, "color": "white"},
        style_cell={"padding": "8px", "textAlign": "left", "border": f"1px solid {BORDER}"},
        style_table={"overflowX": "auto"},
    )

    ts_records = tearsheet.reset_index().rename(
        columns={"index": "Metric", 0: "Value"}
    ).to_dict("records")

    risk_records = [
        {"Metric": k.replace("_", " ").title(), "Value": str(v)}
        for k, v in risk_report.items()
    ]

    regime_table = []
    if regime_stats is not None and not regime_stats.empty:
        regime_table = regime_stats.reset_index().to_dict("records")

    # ══════════════════════════════════════════════════════════════
    # Layout
    # ══════════════════════════════════════════════════════════════
    app.layout = dbc.Container(fluid=True, style={"backgroundColor": BG, "minHeight": "100vh"}, children=[

        # Header
        dbc.Row(dbc.Col(html.Div([
            html.H1("QuantCore", style={"color": GOLD, "fontWeight": "900",
                                        "fontSize": "2.5rem", "margin": 0}),
            html.P("Institutional-grade quantitative research platform",
                   style={"color": "#aaa", "margin": 0}),
        ], style={"padding": "24px 0 12px 0", "borderBottom": f"2px solid {GOLD}",
                  "marginBottom": "20px"}))),

        # KPIs
        kpis,

        # Section: Performance
        dbc.Row(dbc.Col(html.H4("Portfolio Performance",
                                style={"color": CYAN, "borderLeft": f"4px solid {CYAN}",
                                       "paddingLeft": "10px", "marginBottom": "12px"}))),
        dbc.Row(dbc.Col(_card(dcc.Graph(figure=fig_equity(), config={"displayModeBar": False})))),

        # Section: Rolling metrics + VaR
        dbc.Row([
            dbc.Col(_card(dcc.Graph(figure=fig_rolling(),     config={"displayModeBar": False})), md=7),
            dbc.Col(_card(dcc.Graph(figure=fig_var(),         config={"displayModeBar": False})), md=5),
        ]),

        # Section: Distribution
        dbc.Row(dbc.Col(_card(dcc.Graph(figure=fig_distribution(), config={"displayModeBar": False})))),

        # Section: Regime detection
        dbc.Row(dbc.Col(html.H4("Market Regime Detection (Hidden Markov Model)",
                                style={"color": GREEN, "borderLeft": f"4px solid {GREEN}",
                                       "paddingLeft": "10px", "marginBottom": "12px"}))),
        dbc.Row(dbc.Col(_card(dcc.Graph(figure=fig_regime(), config={"displayModeBar": False})))),

        dbc.Row(dbc.Col(_card([
            html.H5("Regime Statistics", style={"color": GREEN, "marginBottom": "10px"}),
            dash_table.DataTable(
                data=regime_table,
                columns=[{"name": c, "id": c} for c in (
                    regime_stats.reset_index().columns if regime_stats is not None and not regime_stats.empty
                    else []
                )],
                **tbl_style,
            ) if regime_table else html.P("No regime data", style={"color": "#aaa"}),
        ]))),

        # Section: Signal decay
        dbc.Row(dbc.Col(html.H4("Factor Signal Decay Analysis",
                                style={"color": PURPLE, "borderLeft": f"4px solid {PURPLE}",
                                       "paddingLeft": "10px", "marginBottom": "12px"}))),
        dbc.Row(dbc.Col(_card(dcc.Graph(figure=fig_ic_decay(), config={"displayModeBar": False})))),

        # Section: 3D Charts
        dbc.Row(dbc.Col(html.H4("3D Visualizations",
                                style={"color": GOLD, "borderLeft": f"4px solid {GOLD}",
                                       "paddingLeft": "10px", "marginBottom": "12px"}))),

        dbc.Row(dbc.Col(_card(dcc.Graph(figure=fig_3d_frontier(), config={"scrollZoom": True})))),
        dbc.Row(dbc.Col(_card(dcc.Graph(figure=fig_3d_ic_surface(), config={"scrollZoom": True})))),
        dbc.Row(dbc.Col(_card(dcc.Graph(figure=fig_3d_drawdown(), config={"scrollZoom": True})))),

        # Section: Tearsheet + Risk
        dbc.Row(dbc.Col(html.H4("Full Performance Tearsheet & Risk Report",
                                style={"color": ORANGE, "borderLeft": f"4px solid {ORANGE}",
                                       "paddingLeft": "10px", "marginBottom": "12px"}))),
        dbc.Row([
            dbc.Col(_card([
                html.H5("Performance Tearsheet", style={"color": CYAN, "marginBottom": "10px"}),
                dash_table.DataTable(data=ts_records,
                                     columns=[{"name": c, "id": c} for c in ["Metric", "Value"]],
                                     **tbl_style),
            ]), md=6),
            dbc.Col(_card([
                html.H5("Risk Report", style={"color": RED, "marginBottom": "10px"}),
                dash_table.DataTable(data=risk_records,
                                     columns=[{"name": "Metric", "id": "Metric"},
                                              {"name": "Value",  "id": "Value"}],
                                     **tbl_style),
            ]), md=6),
        ], className="mb-5"),
    ])

    return app
