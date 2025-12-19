# Time Series Forecasting Project Report
**Course:** FIN41660 Financial Econometrics
**Date:** December 2025

## 1. Introduction
This project aims to develop an interactive forecasting tool for financial time series data. We utilize Python and the `streamlit` framework to implement and compare three key econometric models: Ordinary Least Squares (OLS), AutoRegressive Integrated Moving Average (ARIMA), and Generalized AutoRegressive Conditional Heteroskedasticity (GARCH).

## 2. Methodology

### 2.1 Data Sources
We sourced daily adjusted close prices for the S&P 500 ETF (SPY) from Yahoo Finance using the `yfinance` library. The data covers the period from Jan 2020 to Jan 2024.
- **Preprocessing:** Calculated Log Returns to ensure stationarity (or approximate) and better statistical properties.
- **Missing Data:** Forward fill was applied to handle minor data gaps.

### 2.2 Models Implemented

#### 2.2.1 OLS (Auto-Regressive)
We implemented a simple AR(k) model using OLS, where the current return is regressed on `k` lagged returns. 
$y_t = \beta_0 + \beta_1 y_{t-1} + \epsilon_t$
Standard errors and t-stats allow us to check for serial correlation.

#### 2.2.2 ARIMA
ARIMA(p,d,q) captures linear dependencies. 
- Auto-Regressive (AR) term $p$
- Integrated (I) term $d$ for differencing
- Moving Average (MA) term $q$
We chose a default specification of (1,1,1) but allowed dynamic selection in the app.

#### 2.2.3 GARCH
While OLS and ARIMA model the conditional mean, GARCH(p,q) models the conditional variance (volatility). This is crucial in finance where "volatility clustering" is observed.
$\sigma^2_t = \omega + \alpha \epsilon^2_{t-1} + \beta \sigma^2_{t-1}$

## 3. Results and Analysis
*(This section should be filled with actual screenshots and numbers from the app)*
- **Data Overview:** The SPY returns show typical financial characteristics—mean near zero, fat tails (kurtosis).
- **OLS Performance:** [Insert OLS R-squared and MSE]
- **ARIMA Performance:** [Insert ARIMA Comparison]
- **GARCH Analysis:** The GARCH model successfully captured periods of high volatility, particularly around market shocks.

## 4. Conclusion
The interactive application successfully allows users to test different model specifications on real-world data. Future improvements could include automated parameter tuning (Auto-ARIMA) and multivariate models (VAR).

## Appendix
- **Code Structure:**
    - `app.py`: UI logic
    - `models.py`: Core algorithms
    - `utils.py`: Data handling
