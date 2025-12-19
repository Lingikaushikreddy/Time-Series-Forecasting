import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils import get_stock_data
from models import TimeSeriesModels
import warnings

warnings.filterwarnings("ignore")

def main():
    print("==========================================")
    print("   Time Series Forecasting Standalone Script")
    print("==========================================")
    
    # Parameters
    ticker = "SPY"
    start_date = "2020-01-01"
    end_date = "2024-01-01"
    
    print(f"Fetching data for {ticker} from {start_date} to {end_date}...")
    df = get_stock_data(ticker, start_date, end_date)
    
    if df.empty:
        print("Error: No data fetched.")
        return

    print(f"Data Loaded. Shape: {df.shape}")
    print(df.head())
    
    # Initialize Models
    ts_models = TimeSeriesModels(df['Log_Return'])
    
    # 1. OLS
    print("\n--- Running OLS Model (Lags=1) ---")
    ols_model, ols_preds, _ = ts_models.run_ols(lags=1)  # Unpacking info
    print("OLS Model Trained.")
    metrics_ols = ts_models.evaluate(ts_models.test, ols_preds)
    print(f"OLS Evaluation: {metrics_ols}")
    
    # 2. ARIMA
    print("\n--- Running ARIMA Model (1,1,1) ---")
    try:
        arima_fit, arima_preds, _ = ts_models.run_arima(order=(1,1,1)) # Unpacking info
        print("ARIMA Model Trained.")
        metrics_arima = ts_models.evaluate(ts_models.test, arima_preds)
        print(f"ARIMA Evaluation: {metrics_arima}")
    except Exception as e:
        print(f"ARIMA Failed: {e}")
        
    # 3. GARCH
    print("\n--- Running GARCH Model (1,1) ---")
    garch_fit, garch_preds = ts_models.run_garch(p=1, q=1)
    print("GARCH Model Trained.")
    print("GARCH Summary Head:")
    print(str(garch_fit.summary())[:500] + "...")
    
    print("\n==========================================")
    print("   Analysis Complete")
    print("==========================================")
    print("This script runs independently as required by the assignment.")

if __name__ == "__main__":
    main()
