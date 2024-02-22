import pandas as pd
from sklearn.preprocessing import LabelEncoder

def prepare_data_for_decision_tree(filename):
    # Load the data into a pandas DataFrame.
    df = pd.read_csv(filename)

    # Drop columns where all values are null
    df = df.dropna(axis=1, how='all')

    # Initialize a label encoder.
    label_encoder = LabelEncoder()

    # Iterate over each column in the DataFrame.
    for column in df.columns:
        # Check if the column name contains 'date' and the data type is not numeric.
        if 'date' in column.lower() and not pd.api.types.is_numeric_dtype(df[column]):
            # Attempt to convert it to datetime.
            df[column] = pd.to_datetime(df[column], errors='coerce')

            # If the conversion was successful and there are non-NaT values,
            # create separate columns for year, month, and day.
            if df[column].notnull().any():
                df['year'] = df[column].dt.year
                df['month'] = df[column].dt.month
                df['day'] = df[column].dt.day
                df[f'{column}_hour'] = df[column].dt.hour
                df[f'{column}_minute'] = df[column].dt.minute
                df[f'{column}_second'] = df[column].dt.second
                df.drop(column, axis=1, inplace=True)  # Drop the original datetime column
        # If the data type of this column is 'object', it is a categorical feature and needs to be converted.
        elif df[column].dtype == 'object':
            # Fill missing values with the string 'missing'.
            df[column].fillna('missing', inplace=True)
            # Convert the column to numerical values using label encoding.
            df[column] = label_encoder.fit_transform(df[column])
        # else:
        #     # If the data type is not 'object', it is a numerical feature.
        #     # Fill missing values with the median of the column.
        #     df[column].fillna(df[column].median(), inplace=True)


    return df