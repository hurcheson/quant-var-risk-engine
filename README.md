# Real-Time Portfolio Value-at-Risk (VaR) & Monte Carlo Simulation Engine

![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)
![Python 3.11](https://img.shields.io/badge/Python-3.11-green.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B.svg)

An industry-grade Quantitative Risk & Portfolio Forecasting System developed to simulate multi-asset portfolio returns under Geometric Brownian Motion (GBM), calculate Value-at-Risk (VaR) and Conditional Value-at-Risk (CVaR / Expected Shortfall), and backtest tail risk.

Developed by **Hurcheson** ([hurchesonfori@gmail.com](mailto:hurchesonfori@gmail.com)).

---

## Key Features

- **Geometric Brownian Motion (GBM) Simulations**: 10,000+ stochastic iterations for multi-asset portfolio loss modeling.
- **VaR & CVaR Tail Risk Analytics**: Calculates 95%/99% 1-day Value-at-Risk and Expected Shortfall.
- **Interactive Streamlit Dashboard**: Real-time parameter tuning and Plotly loss distribution visualizations.
- **Automated Unit Testing**: Includes robust test suite covering return distributions and backtesting metrics.

---

## Quick Start

```bash
# 1. Clone repository
git clone https://github.com/hurcheson/quant-var-risk-engine.git
cd quant-var-risk-engine

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run Monte Carlo Risk Engine CLI
python models/monte_carlo_engine.py

# 4. Launch Interactive Web Dashboard
streamlit run app.py
```

---

## Architecture Overview

```text
├── data/
│   └── fetch_market_data.py      # Market data ingestion & multivariate log-returns
├── models/
│   └── monte_carlo_engine.py     # Stochastic GBM simulation engine, VaR & CVaR
├── app.py                         # Interactive Streamlit dashboard
├── tests/
│   └── test_risk_engine.py       # Automated unit test suite
├── requirements.txt
└── README.md
```

## Author & Citation
- **Author**: Hurcheson
- **GitHub**: [https://github.com/hurcheson](https://github.com/hurcheson)
