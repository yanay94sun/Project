import os
import time
from functools import reduce

import pandas as pd
import numpy as np
import datetime as dt

from typing import Generator, List

import adding_labels

DATE_FORMAT = '%Y/%m/%d'
# DATE_FORMAT = '%m/%d/%Y'


def create_new_empty_df_suitable(original_df: pd.DataFrame, window_size_x: int, window_size_y: int, target_columns_label_name: List[str]) -> pd.DataFrame:
    """
    Creates a new DataFrame skeleton out of the old one.

    Parameters:
    original_df (pandas.DataFrame): The old DataFrame.
    window_size_x (int): Number of entries of the old DF, that will be combined as a single entry in the new DF.
    window_size_y (int): Number of columns in a single entry of the new DF that will be used as label.

    Returns:
    pandas.DataFrame: A new empty DF that is ready to contain several entries of the old DF as a single entry.
    """
    x_cols = [col + str(i + 1) for i in range(window_size_x) for col in original_df.columns]
    y_cols = reduce(lambda x, y: x + y, [[col + str(i) for col in target_columns_label_name] for i in range(window_size_x + 1, window_size_x + window_size_y + 1)])
    columns = x_cols + y_cols
    return pd.DataFrame(columns=columns)


# def generate_riders_date_consistent_data_chunks(df: pd.DataFrame) -> Generator[pd.DataFrame, List[str], None]:
#     """
#     Generate filtered chunks of dataframes, that is fit for entries merge.
#
#     Parameters:
#     df (pandas.DataFrame): Old DataFrame contains all the data.
#     """
#
#     for rider in df["cyclist_id"].unique():
#         rider_df = df[df["cyclist_id"] == rider].sort_values(by='date')
#         unique_dates = rider_df["date"].unique()
#         unique_dates = [dt.datetime.strftime(date, DATE_FORMAT) for date in unique_dates]
#
#         sd_adapter = lambda str_date: dt.datetime.strptime(str_date, DATE_FORMAT)
#         last_start_index = 0
#         for i in range(len(unique_dates) - 1):
#             if sd_adapter(unique_dates[i]) != sd_adapter(unique_dates[i + 1]) - dt.timedelta(days=1):
#                 yield rider_df.iloc[last_start_index:i + 1], unique_dates[last_start_index:i + 1]
#                 last_start_index = i + 1
#         if last_start_index < len(unique_dates) - 1:
#             yield rider_df.iloc[last_start_index:], unique_dates[last_start_index:]

def generate_riders_date_consistent_data_chunks(df: pd.DataFrame) -> Generator[pd.DataFrame, List[str], None]:
    for rider in df['cyclist_id'].unique():
        rider_df = df[df['cyclist_id'] == rider]
        yield rider_df, rider_df['date']


def create_data_frame_for_model(df: pd.DataFrame,
                                window_size_x: int = 3,
                                window_size_y: int = 2,
                                target_labels_column_name=None) -> pd.DataFrame:
    """
    Creates a new DataFrame that is suitable for our model - several entries of the original df as single in the new one.

    Parameters:
    df (pandas.DataFrame): The original DataFrame.
    window_size_x (int): Number of entries in the old DataFrame that will be included as a single entry in the new one.
    window_size_y (int): Number of labels expected to be predicted, as a result of its previous entries.
    target_label_column_name (str): The column name that we wish to predict in our model.

    Returns:
    pandas.DataFrame: A DataFrame that is ready to be inserted into the model.
    """
    if target_labels_column_name is None:
        target_labels_column_name = ["lab"]

    columns = list(df.columns)
    target_column_indexes = [list.index(columns, target_label_column_name) for target_label_column_name in target_labels_column_name]
    for i in range(len(target_column_indexes)):
        target_col_index = target_column_indexes[i]
        columns[-i - 1], columns[target_col_index] = columns[target_col_index], columns[-i - 1]
    df = df[columns]
    new_df = create_new_empty_df_suitable(df, window_size_x, window_size_y, target_labels_column_name)
    for rider_df, unique_dates in generate_riders_date_consistent_data_chunks(df):
        last_first_day_index = len(unique_dates) - window_size_x - window_size_y
        for i, start_date in enumerate(unique_dates):
            if i > last_first_day_index:
                break
            window_x = rider_df.iloc[i:i + window_size_x].values
            window_y = rider_df[target_labels_column_name].iloc[
                       i + window_size_x:i + window_size_x + window_size_y].values

            x = window_x.reshape(len(new_df.columns) - (window_size_y * len(target_labels_column_name)))
            y = window_y.reshape(window_size_y * len(target_labels_column_name))

            new_df.loc[len(new_df)] = np.concatenate((x, y), axis=0)

    return new_df


def play():
    import random

    date = ['2024-01-01', '2024-01-02', '2024-01-03', '2024-01-04', '2024-01-05', '2024-01-06', '2024-01-07',
            '2024-01-08', '2024-01-09', '2024-01-10', '2024-01-11', '2024-01-12', '2024-01-13', '2024-01-14',
            '2024-01-15', '2024-01-16', '2024-01-17', '2024-01-18', '2024-01-19', '2024-01-20']

    date = [dt.datetime.strptime(d, '%Y-%m-%d').strftime('%m/%d/%Y') for d in date]

    data = {
        "date": [],
        "rider": [],
        "aol": [],
        "bol": [],
        "col": [],
        "lab": [],
    }

    for rider in ["r1", "r2", "r3"]:
        for j in range(len(date)):
            data["rider"].append(rider)
            data["date"].append(date[j])
            data["aol"].append(f"a{j}")
            data["bol"].append(f"b{j}")
            data["col"].append(f"c{j}")
            data["lab"].append(random.randint(1, 10))

    df = pd.DataFrame(data)

    print(f"returned:\n{df}")

    return df


if __name__ == '__main__':
    current_dir = os.path.dirname(os.path.abspath(__file__))
    prog_dir = os.path.join(current_dir, "..")
    df1 = pd.read_csv(rf"{prog_dir}/Data/Cleaned_Agg_Workouts_2023.csv")
    df2 = pd.read_csv(rf"{prog_dir}/Data/Cleaned_riderIllnesses.csv")
    df3 = pd.read_csv(rf"{prog_dir}/Data/Cleaned_riderInjuries.csv")
    df_merged1 = adding_labels.merge_selected_columns_from_dfs(df1, df2, ["disrupt", "score"])
    df_merged2 = adding_labels.merge_selected_columns_from_dfs(df1, df3, ["disrupt", "score"])
    for wx in [3, 4, 5, 6, 7, 14, 21]:
        for wy in [1, 2, 3, 7]:
            for df_merged, source_file_identifier_name in [(df_merged1, "Illnesses"), (df_merged2, "Injuries")]:
                t = time.time()
                print(f"creating time series for {source_file_identifier_name} with: x:{wx}, y:{wy}")
                time_series_df = create_data_frame_for_model(
                    df_merged,
                    window_size_x=wx,
                    window_size_y=wy,
                    target_labels_column_name=["disrupt", "score"])
                time_series_df.to_csv(rf"{prog_dir}/Data/AggregatedTimeSeries/{source_file_identifier_name}TimeSeries_x{wx}_y{wy}.csv", index=False)
                print(f"created x:{wx}, y:{wy} in time {time.time() - t}")