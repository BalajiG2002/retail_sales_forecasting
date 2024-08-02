from sklearn.model_selection import GridSearchCV
from xgboost import XGBRegressor

def tune_xgboost(X_train, y_train):
    """
    Perform hyperparameter tuning for XGBoost using GridSearchCV.

    Parameters:
    X_train (pd.DataFrame): Training feature data.
    y_train (pd.Series): Training target data.

    Returns:
    XGBRegressor: Best XGBoost model after tuning.
    """
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [6, 10],
        'learning_rate': [0.01, 0.1]
    }
    
    model = XGBRegressor()
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, scoring='neg_mean_squared_error', cv=3)
    grid_search.fit(X_train, y_train)
    
    print(f"Best parameters: {grid_search.best_params_}")
    return grid_search.best_estimator_
