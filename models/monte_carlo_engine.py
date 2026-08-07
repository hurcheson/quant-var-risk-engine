import numpy as np
import pandas as pd
from scipy.stats import norm

class MonteCarloVaREngine:
    def __init__(self, portfolio_value=1_000_000, confidence_level=0.95, time_horizon_days=1):
        if portfolio_value <= 0:
            raise ValueError("Portfolio value must be positive.")
        if not (0 < confidence_level < 1):
            raise ValueError("Confidence level must be between 0 and 1.")
        if time_horizon_days <= 0:
            raise ValueError("Time horizon must be positive.")

        self.portfolio_value = float(portfolio_value)
        self.confidence_level = float(confidence_level)
        self.time_horizon_days = int(time_horizon_days)

    def simulate_gbm(self, returns_df, weights=None, num_simulations=10000, seed=None):
        if seed is not None:
            np.random.seed(seed)

        num_assets = returns_df.shape[1]
        if weights is None:
            weights = np.ones(num_assets) / num_assets
        else:
            weights = np.array(weights, dtype=float)
            if len(weights) != num_assets:
                raise ValueError(f"Length of weights ({len(weights)}) does not match asset count ({num_assets}).")
            weights_sum = np.sum(weights)
            if weights_sum != 0:
                weights = weights / weights_sum

        portfolio_returns = returns_df.dot(weights)
        mu = portfolio_returns.mean()
        sigma = portfolio_returns.std()

        T = self.time_horizon_days
        # Geometric Brownian Motion simulation over T days:
        # S_T = S_0 * exp((mu - 0.5 * sigma^2) * T + sigma * sqrt(T) * Z)
        z = np.random.normal(0, 1, num_simulations)
        drift = (mu - 0.5 * (sigma ** 2)) * T
        diffusion = sigma * np.sqrt(T) * z
        simulated_returns = np.exp(drift + diffusion) - 1.0

        simulated_portfolio_values = self.portfolio_value * (1 + simulated_returns)
        losses = self.portfolio_value - simulated_portfolio_values

        var_amount = float(np.percentile(losses, self.confidence_level * 100))
        tail_losses = losses[losses >= var_amount]
        cvar_amount = float(tail_losses.mean()) if len(tail_losses) > 0 else var_amount

        return {
            "var_amount": var_amount,
            "cvar_amount": cvar_amount,
            "simulated_losses": losses.tolist(),
            "confidence_level": self.confidence_level,
            "time_horizon_days": self.time_horizon_days,
            "simulations": num_simulations,
            "mean_return": float(mu),
            "volatility": float(sigma)
        }

if __name__ == "__main__":
    from data.fetch_market_data import fetch_synthetic_portfolio_data
    df = fetch_synthetic_portfolio_data()
    returns = df.pct_change().dropna()
    engine = MonteCarloVaREngine(time_horizon_days=1)
    res = engine.simulate_gbm(returns)
    print(f"95% 1-Day Monte Carlo VaR: ${res['var_amount']:,.2f}")
    print(f"95% 1-Day Expected Shortfall (CVaR): ${res['cvar_amount']:,.2f}")

