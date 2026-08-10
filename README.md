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
- **LaTeX Math Rendering & TeX Export**: Interactive `st.latex` formula rendering and downloadable standalone `.tex` risk report generator.
- **Interactive Streamlit Dashboard**: Real-time parameter tuning and Plotly loss distribution visualizations.
- **Automated Unit Testing**: Includes robust test suite covering return distributions, backtesting metrics, and LaTeX export generation.

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

## Mathematical Methodology (LaTeX)

### 1. Geometric Brownian Motion (GBM)
Portfolio asset prices follow stochastic continuous dynamics:
$$dS_t = \mu S_t dt + \sigma S_t dW_t$$

The discrete path solution over time horizon $T$ is given by:
$$S_T = S_0 \exp\left( \left(\mu - \frac{1}{2}\sigma^2\right)T + \sigma \sqrt{T} Z \right), \quad Z \sim \mathcal{N}(0, 1)$$

### 2. Value-at-Risk (VaR)
$$\text{VaR}_{\alpha}(L) = \inf \left\{ l \in \mathbb{R} : P(L > l) \le 1 - \alpha \right\} = \text{Percentile}(L, \alpha)$$

### 3. Conditional Value-at-Risk (CVaR / Expected Shortfall)
$$\text{CVaR}_{\alpha}(L) = \mathbb{E}\left[ L \mid L \ge \text{VaR}_{\alpha}(L) \right]$$

---

## Architecture Overview

```text
├── data/
│   └── fetch_market_data.py      # Market data ingestion & multivariate log-returns
├── models/
│   ├── monte_carlo_engine.py     # Stochastic GBM simulation engine, VaR & CVaR
│   └── latex_report_generator.py # LaTeX formula definitions & .tex document generator
├── app.py                         # Interactive Streamlit dashboard with st.latex & .tex export
├── tests/
│   ├── test_risk_engine.py       # Automated unit test suite for risk engine
│   └── test_latex_export.py      # Unit tests for LaTeX equations & document export
├── requirements.txt
└── README.md
```

## Author & Citation
- **Author**: Hurcheson
- **GitHub**: [https://github.com/hurcheson](https://github.com/hurcheson)

