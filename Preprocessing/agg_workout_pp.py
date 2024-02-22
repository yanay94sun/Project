"""This module is responsible to prepare the aggregated data"""
import os
import pandas as pd
from pathlib import Path

PROJECT_PATH = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_PATH / "Data" / "Agg_Workouts_2023.csv"


def load_data(file_path: Path) -> pd.DataFrame:
    """
    Load the data from the specified file path.

    Parameters:
    file_name (str): The name of the CSV file containing the data.

    Returns:
    pandas.DataFrame: The loaded DataFrame.
    """
    # Load the data
    data = pd.read_csv(file_path)

    return data


def display_data(data: pd.DataFrame, num_rows: int = 5) -> None:
    """
    Display the first few rows of the input DataFrame.

    Parameters:
    data (pandas.DataFrame): The input DataFrame.
    num_rows (int): The number of rows to display.
                    Default: 5.
    """
    # Display the first few rows of the DataFrame
    print(data.head(num_rows))


def handle_missing_values(data, strategy='median'):
    """
    Handle missing values in the DataFrame.

    Parameters:
    data (pandas.DataFrame): The input DataFrame.
    strategy (str): The strategy to use for handling missing values in numerical columns.
                    Options: 'mean', 'median', 'mode'.
                    Default: 'median'.

    Returns:
    pandas.DataFrame: DataFrame with missing values handled.
    """
    # Handle missing values in numerical columns
    numerical_cols = data.select_dtypes(include=['number']).columns
    if strategy == 'mean':
        data[numerical_cols] = data[numerical_cols].fillna(data[numerical_cols].mean())
    elif strategy == 'median':
        data[numerical_cols] = data[numerical_cols].fillna(data[numerical_cols].median())
    elif strategy == 'mode':
        data[numerical_cols] = data[numerical_cols].fillna(data[numerical_cols].mode().iloc[0])
    elif strategy == 'drop':
        data = data.dropna(subset=numerical_cols)
    elif strategy == 'zero':
        data[numerical_cols] = data[numerical_cols].fillna(0)
    else:
        raise ValueError("Invalid strategy. Please choose from 'mean', 'median', or 'mode'.")

    # Handle missing values in categorical columns
    categorical_cols = data.select_dtypes(include=['object']).columns
    data[categorical_cols] = data[categorical_cols].fillna(data[categorical_cols].mode().iloc[0])

    return data


# Load and display the data
data = load_data(DATA_PATH)
display_data(data)
# Clean the data using mean imputation for missing values
cleaned_data = handle_missing_values(data)
# Apply missing value handling using median strategy for numerical columns
data = handle_missing_values(data, strategy='zero')
