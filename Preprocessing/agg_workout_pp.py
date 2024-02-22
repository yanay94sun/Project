"""This module is responsible to prepare the aggregated data"""
import pandas as pd
from pathlib import Path

FILE_NAME = "Agg_Workouts_2023.csv"
PROJECT_PATH = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_PATH / "Data" / FILE_NAME


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


def handle_missing_values(data, strategy='median') -> pd.DataFrame:
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
    data = _handle_numerical_missing_vals(data, strategy)

    # Handle missing values in categorical columns
    data = _handle_categorical_missing_vals(data)

    return data


def _handle_categorical_missing_vals(data: pd.DataFrame) -> pd.DataFrame:
    categorical_cols = data.select_dtypes(include=['object']).columns
    # check if there are any missing values
    if data[categorical_cols].isnull().sum().sum() == 0:
        print("No missing values found in categorical columns.")
    data[categorical_cols] = data[categorical_cols].fillna(data[categorical_cols].mode().iloc[0])
    return data


def _handle_numerical_missing_vals(data: pd.DataFrame, strategy: str) -> pd.DataFrame:
    # Handle missing values in numerical columns
    numerical_cols = data.select_dtypes(include=['number']).columns
    # check if there are any missing values
    if data[numerical_cols].isnull().sum().sum() == 0:
        print("No missing values found in numerical columns.")
    if strategy == 'mean':
        data[numerical_cols] = data[numerical_cols].fillna(data[numerical_cols].mean())
    elif strategy == 'median':
        data[numerical_cols] = data[numerical_cols].fillna(data[numerical_cols].median())
    elif strategy == 'mode':
        data[numerical_cols] = data[numerical_cols].fillna(data[numerical_cols].mode().iloc[0])
    elif strategy == 'drop':
        data.dropna(subset=numerical_cols, inplace=True)
    elif strategy == 'zero':
        data[numerical_cols] = data[numerical_cols].fillna(0)
    else:
        raise ValueError("Invalid strategy. Please choose from 'mean', 'median', or 'mode'.")
    return data


def remove_empty_columns(data: pd.DataFrame) -> pd.DataFrame:
    """
    Remove columns where the entire column is NULL.

    Parameters:
    data (pandas.DataFrame): The input DataFrame.

    Returns:
    pandas.DataFrame: DataFrame with empty columns removed.
    """
    # Drop columns where all values are null
    data = data.dropna(axis=1, how='all')

    return data


def export_data(data: pd.DataFrame, file_name: str) -> None:
    """
    Export the input DataFrame to a CSV file.

    Parameters:
    data (pandas.DataFrame): The input DataFrame.
    file_name (str): The name of the CSV file to export the data to.
    """
    # Export the data to a CSV file
    data.to_csv(file_name, index=False)


def prepare_data(data: pd.DataFrame, strategy='mean') -> pd.DataFrame:
    """
    Prepare the input data for modeling.

    Parameters:
    data (pandas.DataFrame): The input DataFrame.
    strategy (str): The strategy to use for handling missing values in numerical columns.
                    Options: 'mean', 'median', 'mode'.
                    Default: 'mean'.

    Returns:
    pandas.DataFrame: The prepared DataFrame.
    """
    # Load and display the data
    display_data(data)
    # Clean the data using mean imputation for missing values
    cleaned_data = handle_missing_values(data, strategy=strategy)
    # Remove empty columns
    cleaned_data = remove_empty_columns(cleaned_data)
    # Export the cleaned data to a new CSV file
    export_data(cleaned_data, f"{PROJECT_PATH}/Data/Cleaned_{FILE_NAME}")

    return cleaned_data
