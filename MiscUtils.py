from pathlib import Path

import pandas as pd


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
