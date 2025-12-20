# Video Presentation Script
**Duration:** 10 Minutes

## Introduction (0:00 - 1:30)
*   **Speaker 1:** Good morning. Today we present our Financial Forecasting Tool.
*   **Speaker 1:** We built a robust web application using **Streamlit** to model and forecast stock returns.
*   **Speaker 1:** Our goal was to compare three models: OLS, ARIMA, and GARCH, and provide actionable insights for investors.
*   **Speaker 1:** We chose Streamlit for the interface because it allows for rapid interaction with the data.

## Demo: Data Loading (1:30 - 3:00)
*   **Speaker 2:** (Screen share app) As you can see, on the left sidebar, we can select any stock ticker (default SPY) and the date range.
*   **Speaker 2:** When we click "Fetch Data", the app downloads real-time data from Yahoo Finance.
*   **Speaker 2:** On the main screen, we visualize the Raw Price history and summary statistics.

## Demo: OLS and ARIMA (3:00 - 6:00)
*   **Speaker 3:** Now let's look at the models. First, OLS. We use a lagged auto-regressive approach.
*   **Speaker 3:** (Show OLS results) You can see the summary table and our new **Advanced Evaluation Metrics** like MAPE and Directional Accuracy.
*   **Speaker 3:** (Point to plot) The chart now includes a shaded **95% Confidence Interval**, giving us a range of likely outcomes, not just a point estimate.
*   **Speaker 3:** (Expand Diagnostics) If we expand the "Advanced Diagnostics" tab, we can verify our assumptions using the Q-Q plot and ACF charts.
*   **Speaker 3:** Moving to the **Model Comparison Tab**. Here we can see OLS and ARIMA side-by-side. This makes it easy to decide which model performs better on the test set.

## Demo: GARCH (6:00 - 8:00)
*   **Speaker 4:** Finally, the GARCH model. Unlike the others, this forecasts volatility.
*   **Speaker 4:** (Show GARCH plot) You can see spikes in the predicted volatility corresponding to market stress events. This is crucial for risk management.

## Code & Standalone Script (8:00 - 9:00)
*   **Speaker 1:** Per the requirements, we also have a `standalone_script.py` which runs the same analysis in the terminal without the UI.
*   **Speaker 1:** (Show terminal output) It outputs the same metrics (MSE, RMSE, MAE) and **exports them to a CSV file** for full reproducibility.

## Conclusion (9:00 - 10:00)
*   **All:** In conclusion, we successfully applied econometric theory to build a practical tool. We learned about the challenges of real-world data and model fitting. Thank you!
