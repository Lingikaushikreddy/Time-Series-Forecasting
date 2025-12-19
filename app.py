import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.graphics.gofplots import qqplot
from utils import get_stock_data
from models import TimeSeriesModels

# Page Config
st.set_page_config(page_title="Financial Forecasting App", layout="wide", page_icon="📈")

# Styling
st.markdown("""
<style>
    .main {
        background-color: #f5f5f5;
    }
    .stButton>button {
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)

st.title("📈 Time Series Forecasting Application")
st.markdown("Financial Forecasting Project for FIN41660 - **Interactive Forecasting Tool**")

# Sidebar
st.sidebar.header("Configuration")
ticker = st.sidebar.text_input("Stock Ticker", value="SPY")
start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime("2020-01-01"))
end_date = st.sidebar.date_input("End Date", value=pd.to_datetime("2024-01-01"))

# Data Fetching
@st.cache_data
def load_data(ticker, start, end):
    return get_stock_data(ticker, start, end)

if st.sidebar.button("Fetch Data"):
    st.session_state['data_fetched'] = True
    
# Main Data Logic
try:
    df = load_data(ticker, start_date, end_date)
    if df.empty:
        st.error("Error: No data found for this ticker/date range. Please try again.")
        st.stop()
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

# Layout Tabs
tab1, tab2 = st.tabs(["📊 Model Analysis", "⚔️ Model Comparison"])

ts_models = TimeSeriesModels(df['Log_Return'])

with tab1:
    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("Raw Data Price")
        st.line_chart(df['Price'])
    with col2:
        st.subheader("Data Statistics")
        st.dataframe(df.describe())

    st.header("Single Model Analysis")
    model_type = st.selectbox("Select Model", ["OLS", "ARIMA", "GARCH"])

    if model_type == "OLS":
        st.subheader("OLS Model (Auto-Regressive)")
        lags = st.slider("Number of Lags", 1, 10, 1)
        
        if st.button("Run OLS"):
            model, preds, conf_int = ts_models.run_ols(lags=lags)
            st.write(model.summary())
            
            # Plotting
            st.subheader("Forecast vs Actual (Test Set)")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=ts_models.test.index, y=ts_models.test, name="Actual", line=dict(color='blue')))
            fig.add_trace(go.Scatter(x=preds.index, y=preds, name="Forecast", line=dict(color='red')))
            # Confidence Interval
            fig.add_trace(go.Scatter(x=conf_int.index, y=conf_int['upper'], mode='lines', line=dict(width=0), showlegend=False))
            fig.add_trace(go.Scatter(x=conf_int.index, y=conf_int['lower'], mode='lines', line=dict(width=0), fill='tonexty', fillcolor='rgba(255, 0, 0, 0.2)', name='95% CI'))
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Metrics
            metrics = ts_models.evaluate(ts_models.test, preds)
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("MSE", f"{metrics['MSE']:.6f}")
            c2.metric("RMSE", f"{metrics['RMSE']:.6f}")
            c3.metric("MAE", f"{metrics['MAE']:.6f}")
            c4.metric("MAPE", f"{metrics['MAPE']:.2f}%")
            c5.metric("Dir. Acc", f"{metrics['Dir. Acc (%)']:.2f}%")
            
            # Export
            csv = preds.to_csv().encode('utf-8')
            st.download_button("Download Forecast CSV", csv, "ols_forecast.csv", "text/csv")
            
            # --- Advanced Visualizations (OLS) ---
            with st.expander("🔍 Advanced Diagnostics (Residual Analysis)"):
                residuals = model.resid
                
                # 1. Residuals over time
                st.markdown("#### 1. Residuals over Time")
                fig_res = px.line(x=residuals.index, y=residuals, labels={'x': 'Date', 'y': 'Residual'})
                st.plotly_chart(fig_res, use_container_width=True)
                
                # 2. Histogram & Q-Q Plot
                col_d1, col_d2 = st.columns(2)
                with col_d1:
                    st.markdown("#### 2. Histogram of Residuals")
                    fig_hist = px.histogram(residuals, nbins=50, title="Distribution")
                    st.plotly_chart(fig_hist, use_container_width=True)
                with col_d2:
                    st.markdown("#### 3. Q-Q Plot")
                    fig_qq = plt.figure(figsize=(6, 4))
                    ax = fig_qq.add_subplot(111)
                    qqplot(residuals, line='s', ax=ax)
                    st.pyplot(fig_qq)
                    
                # 3. ACF / PACF
                st.markdown("#### 4. Autocorrelation of Residuals")
                col_a1, col_a2 = st.columns(2)
                with col_a1:
                    fig_acf = plt.figure(figsize=(6, 4))
                    plot_acf(residuals, lags=20, ax=fig_acf.gca())
                    st.pyplot(fig_acf)
                with col_a2:
                    fig_pacf = plt.figure(figsize=(6, 4))
                    plot_pacf(residuals, lags=20, ax=fig_pacf.gca())
                    st.pyplot(fig_pacf)

    elif model_type == "ARIMA":
        st.subheader("ARIMA Model")
        c1, c2, c3 = st.columns(3)
        p = c1.number_input("p (AR)", 0, 5, 1)
        d = c2.number_input("d (I)", 0, 2, 1)
        q = c3.number_input("q (MA)", 0, 5, 1)
        
        if st.button("Run ARIMA"):
            try:
                model_fit, forecast, conf_int = ts_models.run_arima(order=(p, d, q))
                st.write(model_fit.summary())
                
                # Plotting
                st.subheader("Forecast vs Actual")
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=ts_models.test.index, y=ts_models.test, name="Actual", line=dict(color='blue')))
                fig.add_trace(go.Scatter(x=forecast.index, y=forecast, name="Forecast", line=dict(color='green')))
                # CI
                fig.add_trace(go.Scatter(x=conf_int.index, y=conf_int['upper'], mode='lines', line=dict(width=0), showlegend=False))
                fig.add_trace(go.Scatter(x=conf_int.index, y=conf_int['lower'], mode='lines', line=dict(width=0), fill='tonexty', fillcolor='rgba(0, 255, 0, 0.2)', name='95% CI'))
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Metrics
                metrics = ts_models.evaluate(ts_models.test, forecast)
                c1, c2, c3, c4, c5 = st.columns(5)
                c1.metric("MSE", f"{metrics['MSE']:.6f}")
                c2.metric("RMSE", f"{metrics['RMSE']:.6f}")
                c3.metric("MAE", f"{metrics['MAE']:.6f}")
                c4.metric("MAPE", f"{metrics['MAPE']:.2f}%")
                c5.metric("Dir. Acc", f"{metrics['Dir. Acc (%)']:.2f}%")
                
                # Export
                csv = forecast.to_csv().encode('utf-8')
                st.download_button("Download Forecast CSV", csv, "arima_forecast.csv", "text/csv")
                
                # --- Advanced Visualizations (ARIMA) ---
                with st.expander("🔍 Advanced Diagnostics (Residual Analysis)"):
                    residuals = model_fit.resid
                    
                    # 1. Residuals over time
                    st.markdown("#### 1. Residuals over Time")
                    fig_res = px.line(x=residuals.index, y=residuals, labels={'x': 'Date', 'y': 'Residual'})
                    st.plotly_chart(fig_res, use_container_width=True)
                    
                    # 2. Histogram & Q-Q Plot
                    col_d1, col_d2 = st.columns(2)
                    with col_d1:
                        st.markdown("#### 2. Histogram of Residuals")
                        fig_hist = px.histogram(residuals, nbins=50, title="Distribution")
                        st.plotly_chart(fig_hist, use_container_width=True)
                    with col_d2:
                        st.markdown("#### 3. Q-Q Plot")
                        fig_qq = plt.figure(figsize=(6, 4))
                        ax = fig_qq.add_subplot(111)
                        qqplot(residuals, line='s', ax=ax)
                        st.pyplot(fig_qq)
                        
                    # 3. ACF / PACF
                    st.markdown("#### 4. Autocorrelation of Residuals")
                    col_a1, col_a2 = st.columns(2)
                    with col_a1:
                        fig_acf = plt.figure(figsize=(6, 4))
                        plot_acf(residuals, lags=20, ax=fig_acf.gca())
                        st.pyplot(fig_acf)
                    with col_a2:
                        fig_pacf = plt.figure(figsize=(6, 4))
                        plot_pacf(residuals, lags=20, ax=fig_pacf.gca())
                        st.pyplot(fig_pacf)
                
            except Exception as e:
                st.error(f"ARIMA Error: {e}")

    elif model_type == "GARCH":
        st.subheader("GARCH Model (Volatility Forecasting)")
        c1, c2 = st.columns(2)
        p_vol = c1.number_input("p (Lag)", 1, 5, 1)
        q_vol = c2.number_input("q (Error)", 1, 5, 1)
        
        if st.button("Run GARCH"):
            model_fit, vol_forecast = ts_models.run_garch(p=p_vol, q=q_vol)
            st.write(model_fit.summary())
            
            st.subheader("Volatility Forecast vs Absolute Returns")
            fig = go.Figure()
            # Proxy for actual volatility
            actual_vol = df['Log_Return'].loc[vol_forecast.index].abs()
            fig.add_trace(go.Scatter(x=actual_vol.index, y=actual_vol, name="|Actual Returns|", opacity=0.3))
            fig.add_trace(go.Scatter(x=vol_forecast.index, y=vol_forecast, name="Predicted Volatility (Sigma)", line=dict(color='orange')))
            st.plotly_chart(fig, use_container_width=True)
            
            # Metrics
            metrics = ts_models.evaluate_garch(vol_forecast)
            st.write("Evaluation against proxy |Returns|:")
            st.json(metrics)

            # Export
            csv = vol_forecast.to_csv().encode('utf-8')
            st.download_button("Download Volatility CSV", csv, "garch_forecast.csv", "text/csv")
            
            # --- Advanced Visualizations (GARCH) ---
            with st.expander("🔍 Advanced Diagnostics (Standardized Residuals)"):
                # For GARCH, we check Standardized Residuals (resid / volatility)
                std_resid = model_fit.std_resid
                
                st.markdown("#### 1. Standardized Residuals over Time")
                fig_res = px.line(x=std_resid.index, y=std_resid, labels={'x': 'Date', 'y': 'Std Residual'})
                st.plotly_chart(fig_res, use_container_width=True)
                
                col_d1, col_d2 = st.columns(2)
                with col_d1:
                    st.markdown("#### 2. Histogram of Std Residuals")
                    fig_hist = px.histogram(std_resid, nbins=50, title="Normal Distribution Check")
                    st.plotly_chart(fig_hist, use_container_width=True)
                with col_d2:
                    st.markdown("#### 3. Q-Q Plot")
                    fig_qq = plt.figure(figsize=(6, 4))
                    ax = fig_qq.add_subplot(111)
                    qqplot(std_resid, line='s', ax=ax)
                    st.pyplot(fig_qq)
                
                st.markdown("#### 4. ACF of Squared Std Residuals")
                st.info("If GARCH is good, there should be NO autocorrelation in squared standardized residuals.")
                fig_acf = plt.figure(figsize=(10, 4))
                plot_acf(std_resid**2, lags=20, ax=fig_acf.gca())
                st.pyplot(fig_acf)

with tab2:
    st.header("Side-by-Side Model Comparison")
    st.markdown("Compare OLS and ARIMA performance on the test set.")
    
    if st.button("Run Comparison"):
        with st.spinner("Running models..."):
            # OLS
            _, ols_preds, _ = ts_models.run_ols(lags=1)
            ols_metrics = ts_models.evaluate(ts_models.test, ols_preds)
            
            # ARIMA
            _, arima_preds, _ = ts_models.run_arima(order=(1,1,1))
            arima_metrics = ts_models.evaluate(ts_models.test, arima_preds)
            
            # Comparison Table
            comp_df = pd.DataFrame([ols_metrics, arima_metrics], index=["OLS (AR1)", "ARIMA (1,1,1)"])
            st.subheader("Error Metrics Comparison")
            st.dataframe(comp_df.style.highlight_min(axis=0, color='lightgreen'))
            
            # Combined Plot
            st.subheader("Forecast Comparison")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=ts_models.test.index, y=ts_models.test, name="Actual", line=dict(color='black', width=1)))
            fig.add_trace(go.Scatter(x=ols_preds.index, y=ols_preds, name="OLS Forecast", line=dict(color='red', dash='dot')))
            fig.add_trace(go.Scatter(x=arima_preds.index, y=arima_preds, name="ARIMA Forecast", line=dict(color='green', dash='dot')))
            st.plotly_chart(fig, use_container_width=True)
