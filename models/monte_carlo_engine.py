import numpy as np
import pandas as pd
from scipy.stats import norm

class MonteCarloVaREngine:
    def __init__(self, portfolio_value=1_000_000, confidence_level=0.95, time_horizon_days=1):
        self.portfolio_value = portfolio_value
        self.confidence_level = confidence_level
        self.time_horizon_days = time_horizon_days

    def simulate_gbm(self, returns_df, weights=None, num_simulations=10000):
        if weights is None:
            weights = np.ones(returns_df.shape[1]) / returns_df.shape[1]
        
        weights = np.array(weights)
        portfolio_returns = returns_df.dot(weights)
        mu = portfolio_returns.mean()
        sigma = portfolio_returns.std()

        # Geometric Brownian Motion simulation
        simulated_returns = np.random.normal(mu, sigma, num_simulations)
        simulated_portfolio_values = self.portfolio_value * (1 + simulated_returns)
        losses = self.portfolio_value - simulated_portfolio_values

        var_amount = np.percentile(losses, self.confidence_level * 100)
        cvar_amount = losses[losses >= var_amount].mean()

        return {
            "var_amount": float(var_amount),
            "cvar_amount": float(cvar_amount),
            "simulated_losses": losses.tolist(),
            "confidence_level": self.confidence_level,
            "simulations": num_simulations
        }

if __name__ == "__main__":
    from data.fetch_market_data import fetch_synthetic_portfolio_data
    df = fetch_synthetic_portfolio_data()
    returns = df.pct_change().dropna()
    engine = MonteCarloVaREngine()
    res = engine.simulate_gbm(returns)
    print(f"95% 1-Day Monte Carlo VaR: ${res['var_amount']:,.2f}")
    print(f"95% 1-Day Expected Shortfall (CVaR): ${res['cvar_amount']:,.2f}")
