from typing import List
import pandas as pd


# def merge_selected_columns_from_dfs(df1: pd.DataFrame, df2: pd.DataFrame, columns_to_merge: List[str]) -> pd.DataFrame:
#     """
#     Merges selected columns from df2 into df1 based on matching 'cyclist_id' (in df1) to 'rider' (in df2)
#     and 'year', 'month', 'day' in both DataFrames. Rows without a match in both DataFrames are discarded.
#
#     Parameters:
#     - df1: pd.DataFrame
#         The first DataFrame to merge into.
#     - df2: pd.DataFrame
#         The second DataFrame from which columns will be merged.
#     - columns_to_merge: List[str]
#         The list of column names from df2 to merge into df1.
#
#     Returns:
#     - pd.DataFrame
#         A new DataFrame with selected columns from df2 merged into df1. Rows without matching keys are discarded.
#     """
#
#     # Ensure the columns to merge also include the key columns with their names as they appear in df2
#     key_columns_df2 = ['rider', 'year', 'month', 'day']
#     all_columns_to_merge = list(set(columns_to_merge + key_columns_df2))
#
#     # Select only the necessary columns from df2
#     df2_selected = df2[all_columns_to_merge]
#
#     # Rename the key columns in df2 to match those in df1 for the merge
#     df2_renamed = df2_selected.rename(columns={'rider': 'cyclist_id'})
#
#     # Merge the DataFrames on the key columns using an inner join
#     merged_df = pd.merge(df1, df2_renamed, on=['cyclist_id', 'year', 'month', 'day'], how='inner')
#
#     return merged_df

def merge_selected_columns_from_dfs(df1: pd.DataFrame, df2: pd.DataFrame, columns_to_merge: List[str]) -> pd.DataFrame:
    """
    Merges selected columns from df2 into df1 based on matching 'cyclist_id' (in df1) to 'rider' (in df2)
    and 'year', 'month', 'day' in both DataFrames. Rows without a match in both DataFrames are discarded.

    Parameters:
    - df1: pd.DataFrame
        The first DataFrame to merge into.
    - df2: pd.DataFrame
        The second DataFrame from which columns will be merged.
    - columns_to_merge: List[str]
        The list of column names from df2 to merge into df1.

    Returns:
    - pd.DataFrame
        A new DataFrame with selected columns from df2 merged into df1. Rows without matching keys are discarded.
    """

    # Ensure the columns to merge also include the key columns with their names as they appear in df2
    key_columns_df2 = ['cyclist_id', 'date']
    all_columns_to_merge = list(set(columns_to_merge + key_columns_df2))

    # Select only the necessary columns from df2
    df2_selected = df2[all_columns_to_merge]

    # Merge the DataFrames on the key columns using an inner join
    merged_df = pd.merge(df1, df2_selected, on=['cyclist_id', 'date'], how='inner')

    return merged_df