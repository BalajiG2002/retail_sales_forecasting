import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(file_path):
    """
    Load sales data from a CSV file.

    Parameters:
    file_path (str): Path to the CSV file.

    Returns:
    pd.DataFrame: Loaded data.
    """
    df = pd.read_csv(file_path, parse_dates=['date'])
    df['date'] = pd.to_datetime(df['date'])
    df.sort_values(by=['date'], inplace=True)
    return df

def split_data(df, target_column='units', test_size=0.2):
    """
    Split the data into training and testing sets.

    Parameters:
    df (pd.DataFrame): The DataFrame containing the data.
    target_column (str): The name of the target column.
    test_size (float): Proportion of the dataset to include in the test split.

    Returns:
    tuple: (X_train, X_test, y_train, y_test)
    """
    X = df.drop(columns=[target_column])
    y = df[target_column]
    return train_test_split(X, y, test_size=test_size, shuffle=False)
