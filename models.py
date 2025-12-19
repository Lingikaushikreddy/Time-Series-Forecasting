import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from arch import arch_model
from sklearn.metrics import mean_squared_error, mean_absolute_error

class TimeSeriesModels:
    """
    Class encapsulating OLS, ARIMA, and GARCH model logic.
    Handles data splitting, training, and evaluation.
    """
    def __init__(self, data):
        """
        Args:
            data (pd.Series): The time series data (typically returns).
        """
        self.data = data
        self.train_size = int(len(data) * 0.8)
        self.train = data.iloc[:self.train_size]
        self.test = data.iloc[self.train_size:]
        
    def evaluate(self, actual, forecast):
        mse = mean_squared_error(actual, forecast)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(actual, forecast)
        
        # MAPE (Mean Absolute Percentage Error) - handling zeros
        # For returns close to 0, MAPE can be huge. We add a small epsilon.
        epsilon = 1e-10
        mape = np.mean(np.abs((actual - forecast) / (actual + epsilon))) * 100
        
        # Directional Accuracy (percentage of correct sign predictions)
        # We compare signs of actual vs forecast
        # np.sign returns -1, 0, 1.
        correct_direction = np.mean(np.sign(actual) == np.sign(forecast)) * 100
        
        return {
            "MSE": mse, 
            "RMSE": rmse, 
            "MAE": mae, 
            "MAPE": mape, 
            "Dir. Acc (%)": correct_direction
        }

    def run_ols(self, lags=1):
        """
        Runs an Auto-Regressive OLS model (AR(p)).
        Returns: model_result, forecast_series, conf_int_df
        """
        df = pd.DataFrame(self.train)
        df.columns = ['y']
        
        # Create lags
        for i in range(1, lags + 1):
            df[f'lag_{i}'] = df['y'].shift(i)
            
        df.dropna(inplace=True)
        
        X = df[[f'lag_{i}' for i in range(1, lags + 1)]]
        X = sm.add_constant(X)
        y = df['y']
        
        model = sm.OLS(y, X).fit()
        
        # Forecast Loop
        predictions = []
        conf_int_lower = []
        conf_int_upper = []
        
        # Get standard error of prediction (approximated by residual std dev)
        std_resid = model.resid.std()
        
        full_data = self.data
        
        for i in range(len(self.test)):
            idx = self.train_size + i
            lag_vals = [1.0] # Constant
            for lag in range(1, lags + 1):
                # Retrieve lagged values from full dataset (simulating 1-step ahead rolling)
                if idx - lag >= 0:
                    lag_vals.append(full_data.iloc[idx - lag])
                else:
                    lag_vals.append(0) # Should not match here due to train size
            
            # Use get_prediction for rigorous Prediction Intervals
            # This accounts for both parameter uncertainty and residual variance
            pred_obj = model.get_prediction(np.array([lag_vals]))
            pred = pred_obj.predicted_mean[0]
            ci = pred_obj.conf_int(alpha=0.05) # 95% CI
            
            predictions.append(pred)
            conf_int_lower.append(ci[0, 0])
            conf_int_upper.append(ci[0, 1])
            
        pred_series = pd.Series(predictions, index=self.test.index)
        conf_df = pd.DataFrame({
            'lower': conf_int_lower,
            'upper': conf_int_upper
        }, index=self.test.index)
            
        return model, pred_series, conf_df

    def run_arima(self, order=(1, 1, 1)):
        """
        Runs ARIMA model.
        Returns: model_fit, forecast_series, conf_int_df
        """
        # Train model
        model = ARIMA(self.train, order=order)
        model_fit = model.fit()
        
        # Forecast
        forecast_res = model_fit.get_forecast(steps=len(self.test))
        forecast = forecast_res.predicted_mean
        conf_int = forecast_res.conf_int(alpha=0.05)
        conf_int.columns = ['lower', 'upper']
        
        forecast.index = self.test.index
        conf_int.index = self.test.index
        
        return model_fit, forecast, conf_int

    def run_garch(self, p=1, q=1):
        """
        Runs GARCH model.
        Returns: model_fit, vol_forecast_series
        """
        # Multipliying by 100 help convergence
        scale = 100
        scaled_train = self.train * scale 
        
        model = arch_model(scaled_train, vol='Garch', p=p, q=q, dist='Normal')
        model_fit = model.fit(disp='off')
        
        # Forecast volatility
        # horizon=len(self.test) means strictly out-of-sample from end of train
        forecasts = model_fit.forecast(horizon=len(self.test), reindex=False)
        
        # Get variance forecast and take sqrt for volatility
        var_forecast = forecasts.variance.values[-1, :]
        vol_forecast = np.sqrt(var_forecast) / scale # Rescale back
        
        # Create Series
        vol_series = pd.Series(vol_forecast, index=self.test.index[:len(vol_forecast)])
        
        return model_fit, vol_series

    def evaluate_garch(self, vol_forecast):
        """
        Evaluates GARCH model by comparing forecasted volatility 
        against the proxy of squared returns in the test set.
        """
        # Proxy for actual volatility: abs(returns) or sqrt(returns^2)
        actual_vol = np.abs(self.test)
        
        mse = mean_squared_error(actual_vol, vol_forecast)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(actual_vol, vol_forecast)
        
        return {"MSE (vs |Ret|)": mse, "RMSE": rmse, "MAE": mae}

