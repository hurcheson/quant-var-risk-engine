import numpy as np
import pandas as pd

def fetch_synthetic_portfolio_data(tickers=['AAPL', 'MSFT', 'GOOGL', 'AMZN'], days=252*2, seed=42):
    np.random.seed(seed)
    returns = np.random.multivariate_normal(
        mean=[0.0008, 0.0007, 0.0006, 0.0009],
        cov=[
            [0.0004, 0.0002, 0.00015, 0.0002],
            [0.0002, 0.00035, 0.00018, 0.00022],
            [0.00015, 0.00018, 0.0003, 0.00017],
            [0.0002, 0.00022, 0.00017, 0.00045]
        ],
        size=days
    )
    df = pd.DataFrame(returns, columns=tickers)
    price_df = 100 * (1 + df).cumprod()
    return price_df

if __name__ == "__main__":
    df = fetch_synthetic_portfolio_data()
    print("Market Data Sample:")
    print(df.head())
