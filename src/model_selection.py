import statsmodels.api as sm
from xgboost import XGBRegressor

def train_arima(y_train):
    """
    Train an ARIMA model.

    Parameters:
    y_train (pd.Series): Training target data.

    Returns:
    sm.tsa.ARIMAResultsWrapper: Trained ARIMA model.
    """
    model = sm.tsa.ARIMA(y_train, order=(5, 1, 0))
    results = model.fit()
    return results

def train_exponential_smoothing(y_train):
    """
    Train an Exponential Smoothing model.

    Parameters:
    y_train (pd.Series): Training target data.

    Returns:
    sm.tsa.ExponentialSmoothing: Trained Exponential Smoothing model.
    """
    model = sm.tsa.ExponentialSmoothing(y_train, trend='add', seasonal='add', seasonal_periods=7)
    results = model.fit()
    return results

def train_xgboost(X_train, y_train):
    """
    Train an XGBoost model.

    Parameters:
    X_train (pd.DataFrame): Training feature data.
    y_train (pd.Series): Training target data.

    Returns:
    XGBRegressor: Trained XGBoost model.
    """
    model = XGBRegressor(n_estimators=100, max_depth=10, learning_rate=0.01)
    model.fit(X_train, y_train)
    return model
