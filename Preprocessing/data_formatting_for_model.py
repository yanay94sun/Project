import pandas as pd
import numpy as np
import datetime as dt

from typing import Generator, List

DATE_FORMAT = '%Y/%m/%d'
# DATE_FORMAT = '%m/%d/%Y'


def create_new_empty_df_suitable(original_df: pd.DataFrame, window_size_x: int, window_size_y: int) -> pd.DataFrame:
    """
    Creates a new DataFrame skeleton out of the old one.

    Parameters:
    original_df (pandas.DataFrame): The old DataFrame.
    window_size_x (int): Number of entries of the old DF, that will be combined as a single entry in the new DF.
    window_size_y (int): Number of columns in a single entry of the new DF that will be used as label.

    Returns:
    pandas.DataFrame: A new empty DF that is ready to contain several entries of the old DF as a single entry.
    """

    columns = [col + str(i + 1) for i in range(window_size_x) for col in original_df.columns] + \
              [f"label{''.join([str(i) for i in range(window_size_x + 1, window_size_x + window_size_y + 1)])}"]  # -1 in X for the last label which is included in the Y
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
                                target_label_column_name: str = "lab") -> pd.DataFrame:
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
    columns = list(df.columns)
    target_column_index = list.index(columns, target_label_column_name)
    columns[-1], columns[target_column_index] = columns[target_column_index], columns[-1]
    df = df[columns]
    new_df = create_new_empty_df_suitable(df, window_size_x, window_size_y)
    for rider_df, unique_dates in generate_riders_date_consistent_data_chunks(df):
        last_first_day_index = len(unique_dates) - window_size_x - window_size_y
        for i, start_date in enumerate(unique_dates):
            if i > last_first_day_index:
                break
            window_x = rider_df.iloc[i:i + window_size_x].values
            window_y = np.array([np.max(rider_df[target_label_column_name].iloc[
                       i + window_size_x:i + window_size_x + window_size_y].values)])

            x = window_x.reshape(len(new_df.columns) - 1)
            y = window_y

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
    odf = play()
    ndf = create_data_frame_for_model(odf, 3, 2)
    print(ndf)
