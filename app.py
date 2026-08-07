import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
from data.fetch_market_data import fetch_synthetic_portfolio_data
from models.monte_carlo_engine import MonteCarloVaREngine

st.set_page_config(page_title="Monte Carlo Portfolio Risk Engine", layout="wide")

st.title("Real-Time Portfolio Value-at-Risk (VaR) & Monte Carlo Engine")
st.caption("Industry Quantitative Risk Management & Forecasting System | Developed by Hurcheson")

st.sidebar.header("Simulation Parameters")
portfolio_val = st.sidebar.number_input("Portfolio Value ($)", value=1000000, step=50000)
conf_level = st.sidebar.slider("Confidence Level", min_value=0.90, max_value=0.99, value=0.95, step=0.01)
num_sims = st.sidebar.select_slider("Simulations", options=[1000, 5000, 10000, 50000], value=10000)

@st.cache_data
def get_data():
    return fetch_synthetic_portfolio_data()

data_df = get_data()
returns_df = data_df.pct_change().dropna()

engine = MonteCarloVaREngine(portfolio_value=portfolio_val, confidence_level=conf_level)
sim_res = engine.simulate_gbm(returns_df, num_simulations=num_sims)

col1, col2, col3 = st.columns(3)
col1.metric("Portfolio Value", f"${portfolio_val:,.2f}")
col2.metric(f"Monte Carlo VaR ({int(conf_level*100)}%)", f"${sim_res['var_amount']:,.2f}", delta="-Risk Exposure", delta_color="inverse")
col3.metric(f"Expected Shortfall (CVaR)", f"${sim_res['cvar_amount']:,.2f}", delta="-Tail Loss", delta_color="inverse")

st.subheader("Simulated Portfolio Loss Distribution")
fig = px.histogram(sim_res['simulated_losses'], nbins=50, title="Monte Carlo Loss Frequency Distribution ($)")
st.plotly_chart(fig, use_container_width=True)

st.success("Monte Carlo Portfolio Risk Engine Execution Complete.")
