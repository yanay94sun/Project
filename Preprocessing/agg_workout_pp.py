"""This module is responsible to prepare the aggregated data"""
from typing import List
import numpy as np
import pandas as pd
from pandas.api.indexers import BaseIndexer
from pathlib import Path

from MiscUtils import load_data

FILE_NAME = "Agg_Workouts_2023.csv"
PROJECT_PATH = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_PATH / "Data" / FILE_NAME


class BacwardForwardIndexer(BaseIndexer):
    """
    Indexer for rolling window with backward and forward-looking values.
    df = pd.DataFrame({"values": [0, 1, np.nan, 3, 4]})
    indexer = BacwardForwardIndexer(window_size=3)
    df.rolling(indexer).sum()
    --> 
        values
    0     1.0
    1     4.0
    2     8.0
    3     8.0
    4     8.0
    """
    def get_window_bounds(self, num_values, min_periods, center, closed, step):
        start = np.empty(num_values, dtype=np.int64)
        end = np.empty(num_values, dtype=np.int64)
        for i in range(num_values):
            start[i] = i - self.window_size
            end[i] = i + self.window_size
        return start, end


def rolling_mean(data: pd.DataFrame, columns: str | List[str], window_size: int) -> pd.DataFrame:
    """
    Fill missing values in the DataFrame using rolling mean imputation.
    """
    def rolling_mean_per_cyclist(cyclist: int):
        indexer = BacwardForwardIndexer(window_size=window_size)
        rider_col = data[data['cyclist_id'] == cyclist][columns]
        rolled_mean = rider_col.rolling(indexer, min_periods=1).mean()
        # fill na values with rolling mean
        data.loc[data['cyclist_id'] == cyclist, columns] = data.loc[data['cyclist_id'] == cyclist, columns].fillna(rolled_mean)

    for rider in data['cyclist_id'].unique():
        while data[data['cyclist_id'] == rider][columns].isna().sum().sum() > 0:
            rolling_mean_per_cyclist(rider)

    return data


def fix_date(data: pd.DataFrame) -> pd.DataFrame:
    data['workout_datetime'] = pd.to_datetime(data['workout_datetime'])
    data['date'] = data['workout_datetime'].dt.date
    return data


def drop_cols_and_all_null(data: pd.DataFrame) -> pd.DataFrame:
    """
    Drop columns that are not needed and remove columns that are all null.
    """
    data.dropna(axis=1, how='all', inplace=True)
    cols_to_drop = ["workout_title", "workout_type", "workout_id", "workout_tp_id"]
    data.drop(columns=cols_to_drop, inplace=True)
    return data


def handle_missing_vals(data: pd.DataFrame) -> pd.DataFrame:
    """
    Handle missing values in the DataFrame.
    """
    data.loc[:, 'tss_calculation_method'] = data['tss_calculation_method'].fillna("Undefined")

    fill_w_zeros = ["elevation_gain", "elevation_loss", "elevation_average", "elevation_maximum",
                    "elevation_minimum", "total_time", "distance", "calories", "IF", "tss_actual"]

    cols_to_roll = ["temp_avg", "temp_max", "temp_min"]

    for col in fill_w_zeros:
        data.loc[:, col] = data[col].fillna(0)

    data = rolling_mean(data, cols_to_roll, window_size=4)

    return data


def aggregate_workouts(data: pd.DataFrame) -> pd.DataFrame:
    """
    Some workouts occur on the same day, so we need to aggregate them first.
    Note: The data must be clean before passing it to this function.
    """
    agg_data = data.groupby(['cyclist_id', 'date']).agg(
        {
            'workout_week': 'first',
            'workout_month': 'first',
            'elevation_gain': 'sum',
            'elevation_loss': 'sum',
            'elevation_average': 'mean',
            'elevation_maximum': 'max',
            'elevation_minimum': 'min',
            'temp_avg': 'mean',
            'temp_min': 'min',
            'temp_max': 'max',
            'total_time': 'sum',
            'distance': 'sum',
            'calories': 'sum',
            'IF': 'mean',
            'tss_actual': 'sum',
            'tss_calculation_method': 'first'
        }
    ).reset_index()
    return agg_data


def add_missing_days(data: pd.DataFrame) -> pd.DataFrame:
    """
    Add missing days per cyclist to the data.
    """
    def add_missing_days_per_cyclist(rider: int) -> pd.DataFrame:
        rider_dates = data[data['cyclist_id'] == rider]['date']
        date_range = pd.date_range(start=rider_dates.min(), end=rider_dates.max()).date
        complete_df = pd.DataFrame({'date': date_range,
                                    'cyclist_id': [rider] * len(date_range)})
        merged_df = pd.merge(complete_df, data, on=['date', 'cyclist_id'], how='left')
        return merged_df

    return pd.concat([add_missing_days_per_cyclist(cyclist) for cyclist in data['cyclist_id'].unique()]).reset_index(drop=True)


def fill_week_and_month(data: pd.DataFrame) -> pd.DataFrame:
    """
    Fill the week and month columns in the DataFrame.
    """
    dates = pd.to_datetime(data['date'])
    data['workout_week'] = dates.dt.isocalendar().week
    data['workout_month'] = dates.dt.month
    return data


def prepare_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare the input data for modeling.
    """
    data = drop_cols_and_all_null(data)
    data = fix_date(data)
    data = handle_missing_vals(data)
    data = aggregate_workouts(data)
    data = add_missing_days(data)
    data = fill_week_and_month(data)
    data = handle_missing_vals(data)
    return data


if __name__ == "__main__":
    # Load the data
    df = load_data(DATA_PATH)
    df = prepare_data(df)
    # export the cleaned data to a new CSV file
    df.to_csv(PROJECT_PATH / "Data" / "Cleaned_Agg_Workouts_2023.csv", index=False)
