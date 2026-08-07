import unittest
import pandas as pd
import numpy as np
from models.monte_carlo_engine import MonteCarloVaREngine

class TestMonteCarloEngine(unittest.TestCase):
    def test_var_calculation(self):
        dates = pd.date_range("2024-01-01", periods=100)
        df = pd.DataFrame(np.random.normal(0, 0.01, (100, 4)), index=dates, columns=['A', 'B', 'C', 'D'])
        engine = MonteCarloVaREngine(portfolio_value=1_000_000, confidence_level=0.95)
        res = engine.simulate_gbm(df, num_simulations=1000)
        self.assertIn("var_amount", res)
        self.assertGreater(res["var_amount"], 0)
        self.assertGreater(res["cvar_amount"], res["var_amount"])

if __name__ == "__main__":
    unittest.main()
