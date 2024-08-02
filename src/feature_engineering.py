import pandas as pd

def create_features(df):
    """
    Create features from the raw data for model training.

    Parameters:
    df (pd.DataFrame): The DataFrame containing the raw data.

    Returns:
    tuple: (X, y)
    """
    # Convert 'date' to datetime
    df['date'] = pd.to_datetime(df['date'])
    
    # Extract temporal features
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['day_of_week'] = df['date'].dt.dayofweek
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)

    # Create additional features
    df['ad_spend_per_unit'] = df['ad_spend'] / (df['units'] + 1)  # Avoid division by zero
    df['revenue_per_unit'] = df['orderedrevenueamount'] / (df['units'] + 1)  # Avoid division by zero

    # Lag features
    df['units_lag_1'] = df['units'].shift(1)
    df['units_lag_7'] = df['units'].shift(7)
    
    # Rolling statistics
    df['rolling_mean_7'] = df['units'].rolling(window=7).mean()
    df['rolling_std_7'] = df['units'].rolling(window=7).std()
    
    # Drop rows with NaN values created by shifting
    df.dropna(inplace=True)
    
    # Define target and features
    X = df.drop(columns=['units', 'date', 'item_name', 'anarix_id'])
    y = df['units']
    
    return X, y
