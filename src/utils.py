# Placeholder for utility functions
import pandas as pd
import matplotlib.pyplot as plt

def save_predictions(predictions_df, file_path):
    """
    Save the predictions DataFrame to a CSV file.

    Parameters:
    predictions_df (pd.DataFrame): DataFrame containing the predictions.
    file_path (str): Path where the CSV file will be saved.
    """
    predictions_df.to_csv(file_path, index=False)
    print(f"Predictions saved to {file_path}")

def plot_forecast(actual, forecast, model_name):
    """
    Plot actual vs. forecasted values.

    Parameters:
    actual (pd.Series): Actual values.
    forecast (pd.Series or np.array): Forecasted values.
    model_name (str): Name of the model for labeling the plot.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(actual.index, actual, label='Actual', color='blue')
    plt.plot(actual.index, forecast, label=f'{model_name} Prediction', color='red', linestyle='--')
    plt.title(f'Actual vs {model_name} Forecast')
    plt.xlabel('Date')
    plt.ylabel('Units Sold')
    plt.legend()
    plt.grid(True)
    plt.show()

def evaluate_model(actual, forecast):
    """
    Calculate and return Mean Squared Error between actual and forecasted values.

    Parameters:
    actual (pd.Series): Actual values.
    forecast (pd.Series or np.array): Forecasted values.

    Returns:
    float: Mean Squared Error (MSE) of the forecast.
    """
    mse = ((actual - forecast) ** 2).mean()
    return mse
