import unittest
import pandas as pd
import numpy as np
from models.monte_carlo_engine import MonteCarloVaREngine
from data.fetch_market_data import fetch_synthetic_portfolio_data

class TestMonteCarloEngine(unittest.TestCase):
    def setUp(self):
        dates = pd.date_range("2024-01-01", periods=100)
        np.random.seed(42)
        self.df_returns = pd.DataFrame(
            np.random.normal(0.0005, 0.01, (100, 4)),
            index=dates,
            columns=['AAPL', 'MSFT', 'GOOGL', 'AMZN']
        )

    def test_var_and_cvar_calculation(self):
        engine = MonteCarloVaREngine(portfolio_value=1_000_000, confidence_level=0.95, time_horizon_days=1)
        res = engine.simulate_gbm(self.df_returns, num_simulations=5000, seed=42)
        self.assertIn("var_amount", res)
        self.assertIn("cvar_amount", res)
        self.assertGreaterEqual(res["cvar_amount"], res["var_amount"])

    def test_custom_weights(self):
        engine = MonteCarloVaREngine(portfolio_value=500_000, confidence_level=0.99)
        weights = [0.4, 0.3, 0.2, 0.1]
        res = engine.simulate_gbm(self.df_returns, weights=weights, num_simulations=1000, seed=42)
        self.assertEqual(res["simulations"], 1000)

    def test_weights_mismatch_raises_error(self):
        engine = MonteCarloVaREngine()
        invalid_weights = [0.5, 0.5]
        with self.assertRaises(ValueError):
            engine.simulate_gbm(self.df_returns, weights=invalid_weights)

    def test_multi_day_time_horizon(self):
        engine_1day = MonteCarloVaREngine(portfolio_value=1_000_000, confidence_level=0.95, time_horizon_days=1)
        engine_10day = MonteCarloVaREngine(portfolio_value=1_000_000, confidence_level=0.95, time_horizon_days=10)
        res_1day = engine_1day.simulate_gbm(self.df_returns, num_simulations=10000, seed=42)
        res_10day = engine_10day.simulate_gbm(self.df_returns, num_simulations=10000, seed=42)
        # 10-day risk should generally exceed 1-day risk
        self.assertGreater(res_10day["var_amount"], res_1day["var_amount"])

    def test_invalid_init_parameters(self):
        with self.assertRaises(ValueError):
            MonteCarloVaREngine(portfolio_value=-100)
        with self.assertRaises(ValueError):
            MonteCarloVaREngine(confidence_level=1.5)
        with self.assertRaises(ValueError):
            MonteCarloVaREngine(time_horizon_days=0)

    def test_fetch_synthetic_portfolio_data_dynamic_tickers(self):
        df_default = fetch_synthetic_portfolio_data()
        self.assertEqual(list(df_default.columns), ['AAPL', 'MSFT', 'GOOGL', 'AMZN'])
        
        custom_tickers = ['NVDA', 'TSLA']
        df_custom = fetch_synthetic_portfolio_data(tickers=custom_tickers)
        self.assertEqual(list(df_custom.columns), custom_tickers)

if __name__ == "__main__":
    unittest.main()

