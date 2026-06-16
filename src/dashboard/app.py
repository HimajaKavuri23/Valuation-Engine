"""
Valuation Engine — Professional Dashboard
------------------------------------------
Dark slate UI with teal and amber accents.
Clean, data-dense, professional.
"""

import sys
sys.path.append(".")

import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from src.data.fetcher import FinancialDataFetcher
from src.models.dcf import DCFModel, DCFAssumptions
from src.models.monte_carlo import MonteCarloSimulator, SimulationConfig

st.set_page_config(
    page_title="Valuation Engine",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&family=DM+Serif+Display:ital@0;1&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}
.stApp {
    background: #0F1117;
    color: #E8EAF0;
}
section[data-testid="stSidebar"] {
    background: #161820;
    border-right: 1px solid rgba(255,255,255,0.06);
}
section[data-testid="stSidebar"] * {
    color: #E8EAF0 !important;
}
header[data-testid="stHeader"] {
    background: #0F1117 !important;
    border-bottom: 1px solid rgba(255,255,255,0.06) !important;
}
.block-container {
    padding: 4rem 2.5rem 2rem;
    max-width: 100%;
}
h1 {
    font-family: 'DM Serif Display', serif !important;
    font-size: 2rem !important;
    color: #E8EAF0 !important;
    letter-spacing: -0.02em !important;
    font-weight: 400 !important;
}
h2, h3 {
    font-family: 'Inter', sans-serif !important;
    font-size: 0.65rem !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
    color: #6B7280 !important;
    font-weight: 500 !important;
}
.metric-card {
    background: #161820;
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 10px;
    padding: 1rem 1.25rem;
}
.metric-label {
    font-size: 0.62rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #6B7280;
    margin-bottom: 0.4rem;
}
.metric-value {
    font-size: 1.35rem;
    font-weight: 500;
    color: #E8EAF0;
    letter-spacing: -0.02em;
}
.metric-value--teal { color: #2DD4BF; }
.metric-delta {
    font-size: 0.7rem;
    margin-top: 0.2rem;
    font-weight: 500;
}
.delta-neg { color: #F87171; }
.delta-pos { color: #2DD4BF; }
.delta-neu { color: #6B7280; }
.verdict-over {
    display: inline-block;
    font-size: 0.65rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    font-weight: 600;
    padding: 0.3rem 0.85rem;
    border-radius: 100px;
    background: rgba(239,68,68,0.12);
    color: #F87171;
    border: 1px solid rgba(239,68,68,0.2);
}
.verdict-under {
    display: inline-block;
    font-size: 0.65rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    font-weight: 600;
    padding: 0.3rem 0.85rem;
    border-radius: 100px;
    background: rgba(45,212,191,0.12);
    color: #2DD4BF;
    border: 1px solid rgba(45,212,191,0.2);
}
.company-name {
    font-family: 'DM Serif Display', serif;
    font-size: 1.8rem;
    color: #E8EAF0;
    letter-spacing: -0.02em;
    line-height: 1.1;
}
.company-meta {
    font-size: 0.68rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #6B7280;
    margin-top: 0.3rem;
}
.section-label {
    font-size: 0.62rem;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: #6B7280;
    margin-bottom: 0.75rem;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid rgba(255,255,255,0.06);
}
[data-testid="collapsedControl"] {
    color: #2DD4BF !important;
    background-color: rgba(45,212,191,0.15) !important;
    border-radius: 50% !important;
}
[data-testid="collapsedControl"] svg {
    fill: #2DD4BF !important;
    stroke: #2DD4BF !important;
}
[data-testid="stSidebarCollapsedControl"] {
    color: #2DD4BF !important;
}
[data-testid="stSidebarCollapsedControl"] svg {
    fill: #2DD4BF !important;
    stroke: #2DD4BF !important;
}
button[aria-label="Open sidebar"] svg {
    fill: #2DD4BF !important;
    stroke: #2DD4BF !important;
}
button[aria-label="Close sidebar"] svg {
    fill: #2DD4BF !important;
    stroke: #2DD4BF !important;
}
button[data-testid="stExpandSidebarButton"] {
    background-color: rgba(45,212,191,0.15) !important;
    border-radius: 50% !important;
}
button[data-testid="stExpandSidebarButton"] span {
    color: #2DD4BF !important;
}
button[data-testid="stExpandSidebarButton"] span[data-testid="stIconMaterial"] {
    color: #2DD4BF !important;
}
.stButton > button {
    background: #2DD4BF !important;
    color: #0F1117 !important;
    border: none !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.72rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    border-radius: 6px !important;
    padding: 0.6rem 1.5rem !important;
}
.stButton > button:hover {
    background: #14B8A6 !important;  
}
div[data-testid="stTextInput"] input {
    background: #0F1117 !important;
    color: #E8EAF0 !important;
    caret-color: #E8EAF0 !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 6px !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.85rem !important;
    letter-spacing: 0.08em !important;
}
div[data-testid="stTextInput"] input:focus {
    border-color: #2DD4BF !important;
    box-shadow: none !important;
}
div[data-testid="stTextInput"] label {
    font-size: 0.62rem !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
    color: #6B7280 !important;
}
div[data-testid="stMetric"] {
    background: #161820;
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 10px;
    padding: 1rem 1.25rem;
}
div[data-testid="stMetric"] label {
    font-size: 0.62rem !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    color: #6B7280 !important;
}
div[data-testid="stMetric"] [data-testid="stMetricValue"] {
    font-size: 1.3rem !important;
    color: #E8EAF0 !important;
    font-weight: 500 !important;
}
.footer-bar {
    margin-top: 3rem;
    padding-top: 1rem;
    border-top: 1px solid rgba(255,255,255,0.06);
    display: flex;
    justify-content: space-between;
    font-size: 0.65rem;
    color: #4B5563;
    letter-spacing: 0.05em;
}
</style>
""", unsafe_allow_html=True)

CHART_BG    = "#161820"
CHART_GRID  = "rgba(255,255,255,0.05)"
TEAL        = "#2DD4BF"
AMBER       = "#FBB024"
INDIGO      = "#6366F1"
RED         = "#F87171"
MUTED       = "#6B7280"
TEXT        = "#E8EAF0"

def chart_layout(height=320, show_legend=True):
    return dict(
        height=height,
        margin=dict(l=8, r=8, t=12, b=8),
        paper_bgcolor=CHART_BG,
        plot_bgcolor=CHART_BG,
        font=dict(family="Inter", color=MUTED, size=11),
        showlegend=show_legend,
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02,
            font=dict(size=10, color=MUTED),
            bgcolor="rgba(0,0,0,0)",
        ),
        xaxis=dict(
            gridcolor=CHART_GRID, linecolor=CHART_GRID,
            tickfont=dict(size=10, color=MUTED),
            zeroline=False,
        ),
        yaxis=dict(
            gridcolor=CHART_GRID, linecolor=CHART_GRID,
            tickfont=dict(size=10, color=MUTED),
            zeroline=False,
        ),
    )


# ── Sidebar ──
with st.sidebar:
    st.markdown("""
    <div style='padding: 0.5rem 0 1.5rem; border-bottom: 1px solid rgba(255,255,255,0.06); margin-bottom: 1.5rem;'>
        <div style='font-family: DM Serif Display, serif; font-size: 1.2rem; color: #E8EAF0; letter-spacing: -0.01em;'>
            Valuation<span style='color:#2DD4BF'>Engine</span>
        </div>
        <div style='font-size: 0.6rem; letter-spacing: 0.1em; text-transform: uppercase; color: #6B7280; margin-top: 0.3rem;'>
            DCF · Monte Carlo · Sensitivity
        </div>
    </div>
    """, unsafe_allow_html=True)

    ticker = st.text_input(
        "TICKER SYMBOL",
        value="AAPL",
        help="Any US stock e.g. MSFT, GOOGL, JPM, TSLA"
    ).upper()

    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
    st.markdown("<div class='section-label'>Growth assumptions</div>", unsafe_allow_html=True)

    revenue_growth = st.slider(
        "Revenue growth rate",
        min_value=0.01, max_value=0.30,
        value=0.08, step=0.01,
        format="%.0f%%"
    )
    terminal_growth = st.slider(
        "Terminal growth rate",
        min_value=0.01, max_value=0.05,
        value=0.03, step=0.005,
        format="%.1f%%"
    )

    st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
    st.markdown("<div class='section-label'>Discount rate</div>", unsafe_allow_html=True)

    risk_free_rate = st.slider(
        "Risk-free rate",
        min_value=0.01, max_value=0.08,
        value=0.045, step=0.005,
        format="%.1f%%"
    )
    equity_risk_prem = st.slider(
        "Equity risk premium",
        min_value=0.03, max_value=0.09,
        value=0.055, step=0.005,
        format="%.1f%%"
    )

    st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
    st.markdown("<div class='section-label'>Simulation</div>", unsafe_allow_html=True)

    n_simulations = st.select_slider(
        "Monte Carlo runs",
        options=[1000, 2000, 5000, 10000],
        value=5000
    )

    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
    run_button = st.button("Run Analysis", use_container_width=True)


# ── Landing state ──
if not run_button:
    st.markdown("""
    <div style='padding: 4rem 0 2rem;'>
        <div style='font-family: DM Serif Display, serif; font-size: 2.8rem; color: #E8EAF0; letter-spacing: -0.03em; line-height: 1.1; max-width: 600px;'>
            Intelligent financial<br>valuation engine
        </div>
        <div style='font-size: 0.88rem; color: #6B7280; line-height: 1.8; max-width: 500px; margin-top: 1.25rem;'>
            DCF modeling, Monte Carlo simulation, and sensitivity analysis
            for any US public company. Enter a ticker and run the analysis.
        </div>
        <div style='margin-top: 2rem; display: flex; gap: 0.75rem; flex-wrap: wrap;'>
            <span style='font-family: Inter; font-size: 0.65rem; letter-spacing: 0.1em; text-transform: uppercase; padding: 0.3rem 0.85rem; border-radius: 100px; border: 1px solid rgba(255,255,255,0.1); color: #9CA3AF;'>Try AAPL</span>
            <span style='font-family: Inter; font-size: 0.65rem; letter-spacing: 0.1em; text-transform: uppercase; padding: 0.3rem 0.85rem; border-radius: 100px; border: 1px solid rgba(255,255,255,0.1); color: #9CA3AF;'>Try MSFT</span>
            <span style='font-family: Inter; font-size: 0.65rem; letter-spacing: 0.1em; text-transform: uppercase; padding: 0.3rem 0.85rem; border-radius: 100px; border: 1px solid rgba(255,255,255,0.1); color: #9CA3AF;'>Try JPM</span>
            <span style='font-family: Inter; font-size: 0.65rem; letter-spacing: 0.1em; text-transform: uppercase; padding: 0.3rem 0.85rem; border-radius: 100px; border: 1px solid rgba(255,255,255,0.1); color: #9CA3AF;'>Try TSLA</span>
            <span style='font-family: Inter; font-size: 0.65rem; letter-spacing: 0.1em; text-transform: uppercase; padding: 0.3rem 0.85rem; border-radius: 100px; border: 1px solid rgba(255,255,255,0.1); color: #9CA3AF;'>Try GOOGL</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()


# ── Fetch data ──
with st.spinner(f"Fetching live data for {ticker}..."):
    try:
        fetcher  = FinancialDataFetcher(ticker)
        data     = fetcher.get_all_financials()
        overview = data["overview"]
        if overview["name"] == "N/A":
            st.error(f"No data found for '{ticker}'. Please check the ticker symbol.")
            st.stop()
    except Exception as e:
        st.error(f"Error: {e}")
        st.stop()


# ── Run DCF ──
assumptions = DCFAssumptions(
    projection_years = 5,
    revenue_growth   = revenue_growth,
    terminal_growth  = terminal_growth,
    risk_free_rate   = risk_free_rate,
    equity_risk_prem = equity_risk_prem,
)
model   = DCFModel(data, assumptions)
results = model.calculate_intrinsic_value()


# ── Company header ──
col_name, col_verdict = st.columns([3, 1])
with col_name:
    st.markdown(f"""
    <div class='company-name'>{overview['name']}</div>
    <div class='company-meta'>{ticker} · {overview['sector']} · {overview['industry']}</div>
    """, unsafe_allow_html=True)
with col_verdict:
    st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
    if results["upside_downside_pct"] > 0:
        verdict_html = f"<span class='verdict-under'>Undervalued &nbsp; {results['upside_downside_pct']:+.1f}%</span>"
    else:
        verdict_html = f"<span class='verdict-over'>Overvalued &nbsp; {results['upside_downside_pct']:+.1f}%</span>"
    st.markdown(verdict_html, unsafe_allow_html=True)

st.markdown("<div style='height:1.25rem'></div>", unsafe_allow_html=True)


# ── Metrics row ──
m1, m2, m3, m4, m5 = st.columns(5)
with m1:
    st.metric("Current Price", f"${overview['current_price']:.2f}")
with m2:
    delta = f"{results['upside_downside_pct']:+.1f}%"
    st.metric("Intrinsic Value", f"${results['intrinsic_value']:.2f}", delta=delta)
with m3:
    st.metric("Market Cap", f"${overview['market_cap']/1e9:.1f}B")
with m4:
    st.metric("WACC", f"{results['wacc']:.2%}")
with m5:
    fcf = data["fcf"]
    fcf_val = fcf.dropna().iloc[0] if not fcf.empty else 0
    st.metric("Latest FCF", f"${fcf_val/1e9:.1f}B")

st.markdown("<div style='height:1.5rem'></div>", unsafe_allow_html=True)


# ── Charts row 1 ──
col_wf, col_fcf = st.columns(2)

with col_wf:
    st.markdown("<div class='section-label'>DCF value breakdown</div>", unsafe_allow_html=True)
    pv_fcfs  = results["pv_fcfs"]
    pv_term  = results["pv_terminal"]
    debt     = results["total_debt"]
    cash     = results["cash"]

    labels   = [f"FCF Yr {i+1}" for i in range(len(pv_fcfs))] + \
               ["Terminal Value", "Less Debt", "Plus Cash", "Equity Value"]
    values   = pv_fcfs + [pv_term, -debt, cash, results["equity_value"]]
    measures = ["relative"] * (len(pv_fcfs) + 2) + ["relative", "total"]

    fig_wf = go.Figure(go.Waterfall(
        orientation="v",
        measure=measures,
        x=labels,
        y=values,
        connector={"line": {"color": "rgba(255,255,255,0.08)", "width": 1}},
        increasing={"marker": {"color": TEAL}},
        decreasing={"marker": {"color": RED}},
        totals={"marker": {"color": INDIGO}},
    ))
    fig_wf.update_layout(**chart_layout(height=300, show_legend=False))
    fig_wf.update_yaxes(tickformat="$,.0f")
    st.plotly_chart(fig_wf, use_container_width=True)

with col_fcf:
    st.markdown("<div class='section-label'>Free cash flow — historical vs projected</div>",
                unsafe_allow_html=True)
    hist_fcf   = data["fcf"].dropna()
    hist_years = [str(d.year) for d in hist_fcf.index[:4]][::-1]
    hist_vals  = list(hist_fcf.values[:4])[::-1]
    proj_years = [f"Yr {i+1}" for i in range(5)]
    proj_vals  = results["projected_fcfs"]

    fig_fcf = go.Figure()
    fig_fcf.add_trace(go.Bar(
        x=hist_years, y=hist_vals,
        name="Historical",
        marker=dict(color=INDIGO, opacity=0.7),
    ))
    fig_fcf.add_trace(go.Bar(
        x=proj_years, y=proj_vals,
        name="Projected",
        marker=dict(
            color=AMBER, opacity=0.75,
            pattern=dict(shape="/", fgcolor="rgba(251,176,36,0.3)")
        ),
    ))
    fig_fcf.update_layout(**chart_layout(height=300))
    fig_fcf.update_yaxes(tickformat="$,.0f")
    st.plotly_chart(fig_fcf, use_container_width=True)


# ── Monte Carlo ──
st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
st.markdown("<div class='section-label'>Monte Carlo simulation</div>", unsafe_allow_html=True)

with st.spinner(f"Running {n_simulations:,} simulations..."):
    config     = SimulationConfig(n_simulations=n_simulations)
    simulator  = MonteCarloSimulator(data, config)
    mc_results = simulator.run()
    mc_stats   = simulator.get_statistics()

col_hist, col_mc = st.columns([2, 1])

with col_hist:
    fig_mc = go.Figure()
    fig_mc.add_trace(go.Histogram(
        x=mc_results["intrinsic_value"],
        nbinsx=80,
        name="Simulated values",
        marker=dict(color=INDIGO, opacity=0.65),
    ))
    fig_mc.add_vline(
        x=overview["current_price"],
        line=dict(color=RED, width=1.5, dash="dash"),
        annotation=dict(
            text=f"Current ${overview['current_price']:.0f}",
            font=dict(color=RED, size=10),
            yanchor="top",
        )
    )
    fig_mc.add_vline(
        x=mc_stats["median"],
        line=dict(color=TEAL, width=1.5, dash="dot"),
        annotation=dict(
            text=f"Median ${mc_stats['median']:.0f}",
            font=dict(color=TEAL, size=10),
            yanchor="top",
        )
    )
    fig_mc.update_layout(**chart_layout(height=280, show_legend=False))
    fig_mc.update_xaxes(tickformat="$,.0f", title_text="Intrinsic value per share",
                        title_font=dict(size=10, color=MUTED))
    fig_mc.update_yaxes(title_text="Simulations", title_font=dict(size=10, color=MUTED))
    st.plotly_chart(fig_mc, use_container_width=True)

with col_mc:
    st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
    scenarios = {
        "Bear (10th)":  (mc_stats["p10"],  RED),
        "Low (25th)":   (mc_stats["p25"],  MUTED),
        "Base (50th)":  (mc_stats["p50"],  TEXT),
        "High (75th)":  (mc_stats["p75"],  MUTED),
        "Bull (90th)":  (mc_stats["p90"],  TEAL),
    }
    for label, (val, color) in scenarios.items():
        st.markdown(f"""
        <div style='display:flex;justify-content:space-between;align-items:center;
                    padding:0.55rem 0;border-bottom:1px solid rgba(255,255,255,0.04)'>
            <span style='font-size:0.68rem;color:#6B7280;letter-spacing:0.05em'>{label}</span>
            <span style='font-size:0.88rem;font-weight:500;color:{color}'>${val:.2f}</span>
        </div>
        """, unsafe_allow_html=True)

    prob = mc_stats["prob_undervalued"] * 100
    prob_color = TEAL if prob > 50 else RED
    st.markdown(f"""
    <div style='margin-top:0.85rem;background:#0F1117;border-radius:8px;
                padding:0.85rem 1rem;border:1px solid rgba(255,255,255,0.06)'>
        <div style='font-size:0.6rem;letter-spacing:0.1em;text-transform:uppercase;
                    color:#6B7280;margin-bottom:0.4rem'>Probability undervalued</div>
        <div style='font-size:1.4rem;font-weight:500;color:{prob_color}'>{prob:.1f}%</div>
    </div>
    """, unsafe_allow_html=True)


# ── Sensitivity ──
st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
st.markdown("<div class='section-label'>Sensitivity analysis — intrinsic value per share</div>",
            unsafe_allow_html=True)
st.caption("Rows: WACC · Columns: terminal growth rate")

sensitivity = model.sensitivity_analysis()
current     = overview["current_price"]

def color_cell(val):
    try:
        v = float(str(val).replace("$","").replace(",",""))
        if v > current * 1.1:
            return "background-color:#0D2B1F;color:#2DD4BF"
        elif v > current * 0.9:
            return "background-color:#2A2210;color:#FBB024"
        else:
            return "background-color:#1F0E0E;color:#F87171"
    except:
        return "color:#6B7280"

st.dataframe(
    sensitivity.style.map(color_cell),
    use_container_width=True,
    height=300
)
st.caption("Teal = above current price  ·  Amber = within 10%  ·  Red = below current price")


# ── Footer ──
st.markdown(f"""
<div class='footer-bar'>
    <span>Built by Himaja Kavuri · MS Analytics, USC ·
    <a href='https://himajakavuri23.github.io' target='_blank'
       style='color:#2DD4BF;text-decoration:none'>Portfolio</a> ·
    <a href='https://linkedin.com/in/himaja-kavuri' target='_blank'
       style='color:#2DD4BF;text-decoration:none'>LinkedIn</a>
    </span>
    <span>ValuationEngine v1.0 · Data via Yahoo Finance</span>
</div>
""", unsafe_allow_html=True)