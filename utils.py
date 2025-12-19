import yfinance as yf
import pandas as pd
import numpy as np

def get_stock_data(ticker, start_date, end_date):
    """
    Downloads stock data from yfinance.
    
    Args:
        ticker (str): The stock ticker (e.g., 'SPY').
        start_date (str or datetime): Start date.
        end_date (str or datetime): End date.
        
    Returns:
        pd.DataFrame: DataFrame with 'Adj Close' and 'Return'.
    """
    df = yf.download(ticker, start=start_date, end=end_date)
    
    if df.empty:
        return pd.DataFrame()
        
    # Keep only Adj Close if available, else Close
    if 'Adj Close' in df.columns:
        df = df[['Adj Close']]
    elif 'Close' in df.columns:
        df = df[['Close']]
    
    # Flatten multi-level columns if they exist (yfinance sometimes returns MultiIndex)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)
        
    # Rename for consistency
    col_name = df.columns[0]
    df.rename(columns={col_name: 'Price'}, inplace=True)
    
    # Calculate Log Returns (standard for GARCH)
    df['Log_Return'] = np.log(df['Price'] / df['Price'].shift(1))
    
    # Calculate Simple Returns (percentage) for display
    df['Return'] = df['Price'].pct_change() * 100
    
    # Handle missing data (Forward Fill -> Dropna)
    df.ffill(inplace=True)
    df.dropna(inplace=True)
    
    return df
