"""This module is responsible to prepare the rider injuries and rider illness data"""
import pandas as pd
from pathlib import Path

from MiscUtils import remove_empty_columns

PROJECT_PATH = Path(__file__).resolve().parents[1]
INJURIES_FILE_PATH = PROJECT_PATH / "Data" / "riderInjuries.csv"
ILLNESSES_FILE_NAME = PROJECT_PATH / "Data" / "riderIllnesses.csv"


def handle_missing_values(data: pd.DataFrame) -> pd.DataFrame:
    """
    Handle missing values in the DataFrame.
    """
    # Fill disrupt and score columns with 0
    data['disrupt'] = data['disrupt'].fillna(0)
    data['score'] = data['score'].fillna(0)
    return data


def prepare_data(data: pd.DataFrame, agg_data: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare the input data for modeling.
    """
    # Remove empty columns
    data.dropna(axis=1, how='all', inplace=True)
    # chnege column 'disrupt' to int (i.e. "yes" to 1 and "no" to 0)
    data['disrupt'] = data['disrupt'].map({'yes': 1, 'no': 0})
    data['date'] = pd.to_datetime(data['date']).dt.date
    # rename 'rider' column to 'cyclist_id'
    data.rename(columns={'rider': 'cyclist_id'}, inplace=True)
    # change 'cyclist_id' to int64
    data['cyclist_id'] = data['cyclist_id'].astype('int64')
    # sort the data by date and rider
    data = data.sort_values(by=['cyclist_id', 'date']).reset_index(drop=True)
    data = aggregate_data(data)
    data = add_missing_days(data, agg_data)
    data = handle_missing_values(data)
    return data


def aggregate_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate the data by cyclist_id and date.
    """
    # group by cyclist_id and date and sum the values
    agg_data = data.groupby(['cyclist_id', 'date']).agg({
        'score': 'sum',
        'disrupt': 'max'
    }).reset_index()
    return agg_data


def add_missing_days(health_data: pd.DataFrame, agg_workouts: pd.DataFrame) -> pd.DataFrame:
    def add_missing_days_per_cyclist(rider: int) -> pd.DataFrame:
        if rider not in agg_workouts['cyclist_id'].unique():
            return pd.DataFrame()
        rider_dates = agg_workouts[agg_workouts['cyclist_id'] == rider]['date']
        date_range = pd.date_range(start=rider_dates.min(), end=rider_dates.max()).date
        complete_df = pd.DataFrame({'date': date_range,
                                    'cyclist_id': [rider] * len(date_range)})
        merged_df = pd.merge(complete_df, health_data, on=['date', 'cyclist_id'], how='left')
        return merged_df

    return pd.concat(
        [add_missing_days_per_cyclist(cyclist) for cyclist in health_data['cyclist_id'].unique()]).reset_index(
        drop=True)


if __name__ == "__main__":
    # Load the data
    injuries_data = pd.read_csv(INJURIES_FILE_PATH)
    illnesses_data = pd.read_csv(ILLNESSES_FILE_NAME)
    agg_data = pd.read_csv(r"../Data/Cleaned_Agg_Workouts_2023.csv")

    # Prepare the data
    injuries_data = prepare_data(injuries_data, agg_data)
    illnesses_data = prepare_data(illnesses_data, agg_data)

    # Export the cleaned data to new CSV files
    injuries_data.to_csv(PROJECT_PATH / "Data" / "Cleaned_riderInjuries.csv", index=False)
    illnesses_data.to_csv(PROJECT_PATH / "Data" / "Cleaned_riderIllnesses.csv", index=False)
