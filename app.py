import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from data.fetch_market_data import fetch_synthetic_portfolio_data
from models.monte_carlo_engine import MonteCarloVaREngine

st.set_page_config(page_title="Monte Carlo Portfolio Risk Engine", layout="wide", page_icon="📈")

st.title("Real-Time Portfolio Value-at-Risk (VaR) & Monte Carlo Engine")
st.caption("Industry Quantitative Risk Management & Forecasting System | Developed by Hurcheson")

st.sidebar.header("Simulation Parameters")
portfolio_val = st.sidebar.number_input("Portfolio Value ($)", value=1_000_000, step=50_000, min_value=1_000)
conf_level = st.sidebar.slider("Confidence Level", min_value=0.90, max_value=0.99, value=0.95, step=0.01)
time_horizon = st.sidebar.slider("Time Horizon (Days)", min_value=1, max_value=30, value=1, step=1)
num_sims = st.sidebar.select_slider("Simulations", options=[1000, 5000, 10000, 50000], value=10000)

@st.cache_data
def get_data():
    return fetch_synthetic_portfolio_data()

data_df = get_data()
returns_df = data_df.pct_change().dropna()

engine = MonteCarloVaREngine(
    portfolio_value=portfolio_val,
    confidence_level=conf_level,
    time_horizon_days=time_horizon
)
sim_res = engine.simulate_gbm(returns_df, num_simulations=num_sims)

# Summary Key Performance Indicators (KPIs)
col1, col2, col3, col4 = st.columns(4)
col1.metric("Portfolio Value", f"${portfolio_val:,.2f}")
col2.metric(f"Monte Carlo VaR ({int(conf_level*100)}%, {time_horizon}d)", f"${sim_res['var_amount']:,.2f}", delta="-Risk Exposure", delta_color="inverse")
col3.metric(f"Expected Shortfall (CVaR)", f"${sim_res['cvar_amount']:,.2f}", delta="-Tail Loss", delta_color="inverse")
col4.metric("Annualized Volatility", f"{sim_res['volatility'] * np.sqrt(252):.2%}")

st.markdown("---")

tab1, tab2 = st.tabs(["Monte Carlo Simulation & Distribution", "Market Data & Portfolio Analytics"])

with tab1:
    st.subheader("Simulated Portfolio Loss Distribution")
    losses = sim_res['simulated_losses']
    var_val = sim_res['var_amount']
    cvar_val = sim_res['cvar_amount']

    fig_hist = px.histogram(
        losses,
        nbins=60,
        title=f"Monte Carlo Loss Frequency Distribution ({num_sims:,} Simulations over {time_horizon} Day Horizon)",
        labels={'value': 'Loss Amount ($)'},
        color_discrete_sequence=['#1f77b4']
    )
    
    # Add VaR and CVaR vertical threshold lines
    fig_hist.add_vline(x=var_val, line_dash="dash", line_color="orange", annotation_text=f"VaR ({int(conf_level*100)}%): ${var_val:,.2f}", annotation_position="top right")
    fig_hist.add_vline(x=cvar_val, line_dash="solid", line_color="red", annotation_text=f"CVaR: ${cvar_val:,.2f}", annotation_position="top left")
    
    st.plotly_chart(fig_hist, use_container_width=True)

with tab2:
    st.subheader("Historical Market Prices & Returns Analysis")
    col_a, col_b = st.columns(2)
    with col_a:
        fig_price = px.line(data_df, title="Synthetic Historical Asset Prices ($)")
        st.plotly_chart(fig_price, use_container_width=True)
    with col_b:
        corr_matrix = returns_df.corr()
        fig_corr = px.imshow(corr_matrix, text_auto=True, color_continuous_scale="Viridis", title="Asset Returns Correlation Matrix")
        st.plotly_chart(fig_corr, use_container_width=True)

st.success("Monte Carlo Portfolio Risk Engine Execution Complete.")

