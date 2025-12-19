# Time Series Forecasting Project (FIN41660)

## Overview
This project is an interactive Financial Time Series Forecasting tool developed for the FIN41660 Financial Econometrics module. It compares three key econometric models:
- **OLS** (Ordinary Least Squares) - Auto-Regressive implementation.
- **ARIMA** (AutoRegressive Integrated Moving Average).
- **GARCH** (Generalized AutoRegressive Conditional Heteroskedasticity).

The application allows users to forecast stock returns and volatility using real-time data from Yahoo Finance.

## Features
- **Interactive Dashboard:** built with Streamlit for real-time model interaction.
- **Advanced Diagnostics:** Residual analysis, Q-Q plots, and ACF/PACF for model validation.
- **Comparison Dashboard:** Side-by-side comparison of OLS and ARIMA models.
- **Export Capabilities:** Download forecasts and variance results as CSVs.
- **Dynamic Data:** Fetches any stock ticker (SPY, AAPL, etc.) provided by `yfinance`.
- **Model Tuning:** Adjust lags (OLS), p,d,q parameters (ARIMA), and GARCH parameters dynamically.
- **Visualizations:** Interactive plots with 95% Confidence Intervals.
- **Standalone Script:** Reproducible analysis containing MAPE and Directional Accuracy metrics.

## Installation & Quick Start



1.  **Install Python:** Make sure you have Python installed (download from python.org).
2.  **Open in VS Code:** Open this entire folder in VS Code.
3.  **Open Terminal:** In VS Code, go to `Terminal` -> `New Terminal`.
4.  **Install Libraries:** Paste this command and hit Enter:
    ```bash
    pip install -r requirements.txt
    ```
5.  **Run the App:** Once installed, paste this:
    ```bash
    streamlit run app.py
    ```

The app should automatically open in your web browser!

## Usage

### 1. Run the Interactive App
To launch the dashboard in your browser:
### 1. Run the Interactive App
To launch the dashboard in your browser:
```bash
streamlit run app.py
```

### 2. Run the Standalone Script
To run the analysis without the UI (prints metrics to console):
```bash
python standalone_script.py
```

## Project Structure
- `app.py`: Main Streamlit application file.
- `models.py`: Core logic for OLS, ARIMA, and GARCH models.
- `utils.py`: Helper functions for fetching and cleaning financial data.
- `standalone_script.py`: Independent script for reproducibility checks.
- `REPORT_DRAFT.md`: Draft content for the written report.
- `VIDEO_SCRIPT.md`: Script for the video presentation.
- `requirements.txt`: Python package dependencies.

## Models Explained
- **OLS:** Used here as an AR(k) model to predict future returns based on past returns.
- **ARIMA:** Captures linear trends and moving averages in the time series.
- **GARCH:** Models the *volatility* of the series, useful for risk management.

## Contributing

- **Course:** FIN41660, Academic Year 2025/2026
