import os
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier  # Import Decision Tree Classifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split  # Import train_test_split function
from sklearn import metrics  # Import scikit-learn metrics module for accuracy calculation
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import math


seed = 42


def load_data(file_path: Path | str) -> pd.DataFrame:
    """
    Load the data from the specified file path.
    """
    data = pd.read_csv(file_path)
    return data


def merge_selected_columns_from_dfs(df1: pd.DataFrame, df2: pd.DataFrame, columns_to_merge: List[str]) -> pd.DataFrame:
    # Ensure the columns to merge also include the key columns with their names as they appear in df2
    key_columns_df2 = ['cyclist_id', 'date']
    all_columns_to_merge = list(set(columns_to_merge + key_columns_df2))

    # Select only the necessary columns from df2
    df2_selected = df2[all_columns_to_merge]

    # Merge the DataFrames on the key columns using an inner join
    merged_df = pd.merge(df1, df2_selected, on=['cyclist_id', 'date'], how='inner')

    return merged_df


def prepare_data(df: pd.DataFrame, window_size_past: int, window_size_future: int):
    """
    Prepare the data for the model by creating rolling windows of features and labels.
    Example of the output:
    N = window_size_past
    M = window_size_future
    cyclist_id | date | feature1_1 | feature2_1 | disrupt1 | score1 | ... | feature1_N | feature2_N | disruptN | scoreN | label_disrupt1 | label_score1 | ... | label_disruptM | label_scoreM
    """
    tss_method_cols = [col for col in df.columns if "tss_calculation_method" in col]
    df = label_encoding(df, tss_method_cols)
    df = df.apply(pd.to_numeric, errors='coerce')

    feature_cols = df.columns
    remove_keys = ["date", "day", "year", "month", "workout_week", "workout_title", "workout_type", "_id"]
    feature_cols = list(filter(lambda col: not any([key in col for key in remove_keys]), feature_cols))
    prepared_data_list = []
    # Iterate over the unique cyclist_ids to handle each cyclist's data separately
    for cyclist_id in df['cyclist_id'].unique():
        cyclist_data = df[df['cyclist_id'] == cyclist_id]

        rolling_data = {}

        # Create rolling windows of features
        for i in range(1, window_size_past + 1):
            for feature in feature_cols:
                rolling_data[f'{feature}{i}'] = cyclist_data[feature].shift(-i)

        for i in range(1, window_size_future + 1):
            rolling_data[f'label_disrupt{i}'] = cyclist_data['disrupt'].shift(-i - window_size_past)
            rolling_data[f'label_score{i}'] = cyclist_data['score'].shift(-i - window_size_past)

        rolling_data['cyclist_id'] = cyclist_id

        cyclist_rolling_df = pd.DataFrame(rolling_data)
        cyclist_rolling_df = cyclist_rolling_df.dropna()
        prepared_data_list.append(cyclist_rolling_df)

    return pd.concat(prepared_data_list).reset_index(drop=True)


def ROC_curve(classifier, X_test: pd.DataFrame, y_test: pd.DataFrame):
    y_proba = classifier.predict_proba(X_test)
    fpr, tpr, thresholds = metrics.roc_curve(y_test, y_proba[:, 1])
    roc_auc = metrics.auc(fpr, tpr)

    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()


def best_model(classifiers: Dict[str, DecisionTreeClassifier], X_train: pd.DataFrame, y_train: pd.DataFrame,
               X_val: pd.DataFrame, y_val: pd.DataFrame):
    best_auc = 0
    best_classifier = None
    best_threshold = 0
    for name, classifier in classifiers.items():
        y_proba = classifier.predict_proba(X_val)
        fpr, tpr, thresholds = metrics.roc_curve(y_val, y_proba[:, 1])
        roc_auc = metrics.auc(fpr, tpr)
        if roc_auc > best_auc:
            best_auc = roc_auc
            best_classifier = classifier
            best_threshold = thresholds[np.argmax(tpr - fpr)]
    return best_classifier, best_threshold


def evaluate_and_visualize_model(classifier: DecisionTreeClassifier, X_val: pd.DataFrame, y_val: pd.DataFrame,
                                 X_test: pd.DataFrame, y_test: pd.DataFrame, do_print=False):
    y_val_proba = classifier.predict_proba(X_val)
    fpr, tpr, thresholds = metrics.roc_curve(y_val, y_val_proba[:, 1])
    c = thresholds[np.argmax(tpr - fpr)]
    auc = metrics.roc_auc_score(y_val, y_val_proba[:, 1])

    y_test_proba = classifier.predict_proba(X_test)
    y_pred = (y_test_proba[:, 1] >= c).astype(int)

    report = classification_report(y_test, y_pred, output_dict=True)
    report["auc"] = auc

    if do_print:
        print(classification_report(y_val, y_pred))
    return report


def filter_relevant_cols_for_model(df: pd.DataFrame):
    """
    The df looks like this:
    cyclist_id | date | feature1_1 | feature2_1 | disrupt1 | score1 | ... | feature1_N | feature2_N | disruptN | scoreN | label_disrupt1 | label_score1 | ... | label_disruptM | label_scoreM
    This function returns the features and labels that are relevant for the model.
    """
    disrupt_cols = [col for col in df.columns if 'label_disrupt' in col]
    df['label_agg_disrupt'] = df[disrupt_cols].max(axis=1)
    features = [col for col in df.columns if 'label' not in col]
    return features, disrupt_cols
    # return features, ['label_agg_disrupt']
    # return features, disrupt_cols, ['label_agg_disrupt']


def label_encoding(df: pd.DataFrame, column_names: List[str]):
    le = LabelEncoder()
    for column in column_names:
        df[column] = le.fit_transform(df[column])
    return df


def process_report(report: dict):
    res = report['1.0']
    res.pop('support')
    res['accuracy'] = report['accuracy']
    res['auc'] = report['auc']
    return res


def get_train_and_test(prepared_data, window_size_past, window_size_future):
    feature_cols, label_cols = filter_relevant_cols_for_model(prepared_data)
    X = prepared_data[feature_cols]
    y = prepared_data[label_cols]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                        random_state=1)  # 80% training and 10% test
    return X_train, X_test, y_train, y_test


df1 = load_data(rf"Data/Cleaned_Agg_Workouts_2023.csv")
df2 = load_data(rf"Data/Cleaned_riderInjuries.csv")
df_merged = merge_selected_columns_from_dfs(df1, df2, ["disrupt", "score"])
classifier = XGBClassifier(max_depth=3, n_estimators=200, random_state=seed)
window_size_past = 30
window_size_future = 7
prepared_data = prepare_data(df_merged, window_size_past, window_size_future)
X_train, X_test, y_train, y_test = get_train_and_test(prepared_data, window_size_past, window_size_future)
thresholds = None

def train_model():
    global thresholds
    classifier.fit(X_train, y_train)
    thresholds = get_thresholds()

    return classifier


def get_unique_cyclist_ids():
    """
    Get the unique cyclist IDs from the prepared data.
    """
    return df_merged['cyclist_id'].unique().tolist()


def get_thresholds():
    y_test_proba = classifier.predict_proba(X_test)
    num_labels = y_test.shape[1]
    best_thresholds = np.zeros(num_labels)
    for i in range(num_labels):
        fpr, tpr, thresholds = metrics.roc_curve(y_test.iloc[:, i], y_test_proba[:, i])
        best_thresholds[i] = thresholds[np.argmax(tpr - fpr)]
    return best_thresholds

def predict_cyclist_injury_probability(cyclist_id: int):
    """
    Predict the injury probability for the given cyclist ID, based on the trained model and the recent window_size_past data.
    """
    # Get the cyclist data for the given cyclist ID
    cyclist_data = prepared_data[prepared_data['cyclist_id'] == cyclist_id]
    # Get the features for the prepared data
    features, _ = filter_relevant_cols_for_model(cyclist_data)
    # Take the last record from the cyclist data to get the most recent data
    X = cyclist_data[features].tail(1)
    X_values = cyclist_data[features].tail(1).values
    # Predict the injury probability
    injury_probability = classifier.predict_proba(X_values)
    return normalize(injury_probability[0]), process_X(X)


def normalize(values):
    max_vals = thresholds * 2
    normalized = values / max_vals
    return [float(min(1, x)) for x in normalized]

def get_features_for_cyclist(cyclist_id: int):
    """
    Retrieve features for the given cyclist ID along with dates.
    """
    cyclist_data = df_merged[df_merged['cyclist_id'] == cyclist_id].tail(window_size_past)
    feature_dict = {col: cyclist_data[col].round(2).tolist() for col in cyclist_data.columns if col not in ['cyclist_id', 'date']}
    dates = cyclist_data['date'].tolist()
    return feature_dict, dates

def process_X(X: pd.DataFrame):
    d = X.iloc[0].to_dict()
    merged_cols = df_merged.columns
    new_dict = {}
    for col in merged_cols:
        for key in d:
            if col in key:
                new_dict[col] = new_dict.get(col, []) + [round(d[key], 2)]

    return new_dict


# def log_normalize(values):
#     # Apply logarithmic transformation, add a tiny value to avoid log(0)
#     log_values = [math.log(x + 1e-10) for x in values]
#
#     # Find the minimum and maximum of the logarithmic values
#     min_val = min(log_values)
#     max_val = max(log_values)
#
#     # Normalize the logarithmic values to a range of 0 to 1
#     normalized_values = [min(1, ((x - min_val) / (max_val - min_val))) for x in log_values]
#     return normalized_values
