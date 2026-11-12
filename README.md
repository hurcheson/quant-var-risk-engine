# Monte Carlo Portfolio Risk Prototype

A **Python and Streamlit educational prototype** for exploring portfolio-return simulations, Value-at-Risk (VaR), and expected shortfall (CVaR/ES).

**Data source:** generated synthetic asset-price series. **Status:** demonstration code, not a live trading or production risk-management system.

## What it does

1. Generates example asset-price histories from simulated correlated daily returns (`data/fetch_market_data.py`).
2. Forms a weighted portfolio return series, estimates its mean and standard deviation, and draws simulated portfolio returns from a lognormal-horizon approximation (`models/monte_carlo_engine.py`).
3. Calculates empirical loss quantiles (VaR) and average losses in the selected tail (expected shortfall).
4. Displays a simulation summary and loss histogram in Streamlit (`app.py`), with a LaTeX report export (`models/latex_report_generator.py`).
5. Includes unit tests of basic calculations, input validation, and report generation (`tests/`).

The simulation is **portfolio-level**; it does not generate a fully joint set of future asset paths for every security.

## Run locally

Requires a compatible Python environment with the packages listed in `requirements.txt`.

```bash
git clone https://github.com/hurcheson/quant-var-risk-engine.git
cd quant-var-risk-engine
python -m pip install -r requirements.txt

# Console demonstration
python -m models.monte_carlo_engine

# Interactive prototype
streamlit run app.py

# Run the included tests
python -m unittest discover -s tests -v
```

## Mathematical sketch

Let \(V_0\) be the initial portfolio value, \(\hat{\mu}\) and \(\hat{\sigma}\) estimated daily portfolio-return parameters, and \(T\) the simulation horizon measured in days. The code simulates illustrative horizon returns using

\[
R_T = \exp\!\left[(\hat{\mu} - \tfrac12\hat{\sigma}^2)T + \hat{\sigma}\sqrt{T}\,Z\right] - 1, \qquad Z\sim N(0,1).
\]

For simulated loss \(L=-V_0R_T\), the VaR estimate is the chosen empirical loss quantile. The expected shortfall estimate averages simulated losses at or above that quantile.

## Important limitations

- All example price histories are **synthetic**. Company-like ticker labels are placeholders, not evidence of live market feeds.
- The model relies on simple normal/lognormal assumptions and does not capture all tail-risk, dependence, and market-regime behaviors.
- Tests are software checks, **not** out-of-sample financial backtests or external model validation.
- Outputs must not be interpreted as investment advice or operational risk limits.

**Author:** [Hurcheson Asante Ofori](https://github.com/hurcheson) · [Portfolio](https://hurcheson.vercel.app) · [Email](mailto:oforihurcheson@gmail.com)

*This repository does not currently declare a license in a separate LICENSE file; please do not infer a license from older badges or documentation.*
