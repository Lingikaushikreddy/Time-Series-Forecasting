"""
Standalone Script for Financial Time Series Forecasting.
This script runs the analysis pipeline (OLS, ARIMA, GARCH) without the UI.
Useful for debugging and reproducible research.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils import get_stock_data
from models import TimeSeriesModels
from sklearn.metrics import mean_squared_error, mean_absolute_error
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
    
    # Calculate explicit metrics for reproducibility
    print("\n==========================================")
    print("   FINAL METRICS SUMMARY")
    print("==========================================")
    
    # OLS Metrics
    rmse_ols = np.sqrt(mean_squared_error(ts_models.test, ols_preds))
    mae_ols = mean_absolute_error(ts_models.test, ols_preds)
    mse_ols = mean_squared_error(ts_models.test, ols_preds)
    
    print("\nOLS METRICS:")
    print(f"  RMSE: {rmse_ols:.6f}")
    print(f"  MAE:  {mae_ols:.6f}")
    print(f"  MSE:  {mse_ols:.6f}")
    
    # ARIMA Metrics
    try:
        rmse_arima = np.sqrt(mean_squared_error(ts_models.test, arima_preds))
        mae_arima = mean_absolute_error(ts_models.test, arima_preds)
        mse_arima = mean_squared_error(ts_models.test, arima_preds)
        
        print("\nARIMA METRICS:")
        print(f"  RMSE: {rmse_arima:.6f}")
        print(f"  MAE:  {mae_arima:.6f}")
        print(f"  MSE:  {mse_arima:.6f}")
    except:
        print("\nARIMA METRICS: Not available (model failed)")
    
    # Save metrics to CSV for reproducibility
    metrics_df = pd.DataFrame({
        'Model': ['OLS', 'ARIMA'],
        'RMSE': [rmse_ols, rmse_arima if 'rmse_arima' in locals() else np.nan],
        'MAE': [mae_ols, mae_arima if 'mae_arima' in locals() else np.nan],
        'MSE': [mse_ols, mse_arima if 'mse_arima' in locals() else np.nan]
    })
    
    metrics_df.to_csv('metrics_summary.csv', index=False)
    print("\n✓ Metrics saved to 'metrics_summary.csv'")
    
    print("\n==========================================")
    print("   Analysis Complete")
    print("==========================================")
    print("This script runs independently as required by the assignment.")

if __name__ == "__main__":
    main()
