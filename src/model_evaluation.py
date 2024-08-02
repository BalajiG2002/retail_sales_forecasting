import numpy as np
from src.utils import evaluate_model, plot_forecast

def evaluate_models(y_test, arima_pred, es_pred, xgb_pred):
    """
    Evaluate and compare models.

    Parameters:
    y_test (pd.Series): Actual target data.
    arima_pred (np.array): ARIMA model predictions.
    es_pred (np.array): Exponential Smoothing model predictions.
    xgb_pred (np.array): XGBoost model predictions.
    """
    arima_mse = evaluate_model(y_test, arima_pred)
    es_mse = evaluate_model(y_test, es_pred)
    xgb_mse = evaluate_model(y_test, xgb_pred)
    
    print(f'ARIMA Mean Squared Error: {arima_mse}')
    print(f'Exponential Smoothing Mean Squared Error: {es_mse}')
    print(f'XGBoost Mean Squared Error: {xgb_mse}')
    
    # Plot forecasts
    plot_forecast(y_test, arima_pred, 'ARIMA')
    plot_forecast(y_test, es_pred, 'Exponential Smoothing')
    plot_forecast(y_test, xgb_pred, 'XGBoost')
