from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import pandas as pd
from matplotlib import pyplot as plt
from sklearn.tree import DecisionTreeClassifier  # Import Decision Tree Classifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split  # Import train_test_split function
from sklearn import metrics  # Import scikit-learn metrics module for accuracy calculation
from sklearn.metrics import classification_report

current_dir = os.path.dirname(os.path.abspath(__file__))
prog_dir = os.path.join(current_dir, "..")
sys.path.append(prog_dir)
from Preprocessing import adding_labels, data_formatting_for_model


def load_data(file_path: Path | str) -> pd.DataFrame:
    """
    Load the data from the specified file path.
    """
    data = pd.read_csv(file_path)

    return data


def evaluate_and_visualize_model(classifier: DecisionTreeClassifier, X_test: pd.DataFrame, y_test: pd.DataFrame):
    y_pred = classifier.predict(X_test)

    print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    import matplotlib.pyplot as plt
    from sklearn import tree

    # Assuming clf is your trained DecisionTreeClassifier object
    plt.figure(figsize=(12, 8))
    tree.plot_tree(classifier, filled=True, feature_names=X_test.columns)
    plt.show()


def filter_relevant_cols_for_model(df: pd.DataFrame):
    feature_cols = df.columns[:-1]  # removing the label column
    remove_keys = ["date", "day", "year", "month", "workout_week", "workout_title", "workout_type", "tss_cal", "_id"]
    feature_cols = list(filter(lambda col: not any([key in col for key in remove_keys]), feature_cols))
    return feature_cols, [df.columns[-1]]


def create_model(df: pd.DataFrame, classifier, **kwargs):
    feature_cols, label_cols = filter_relevant_cols_for_model(df)
    # feature_cols = df.columns[:-1]
    # label_col = [df.columns[-1]]
    df = df.apply(pd.to_numeric, errors='coerce')
    X = df[feature_cols]
    y = df[label_cols]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4,
                                                        random_state=1)  # 90% training and 10% test

    # clf = DecisionTreeClassifier(max_depth=3, min_samples_split=2, min_samples_leaf=1, max_features=None)
    clf = classifier(**kwargs)
    # Train Decision Tree Classifier
    clf = clf.fit(X_train, y_train)
    evaluate_and_visualize_model(clf, X_test, y_test)
    return clf


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-f",
        default=False,
        type=bool,
        help="create a time series formatted data from the raw data into a csv"
    )

    parser.add_argument(
        "-i",
        default=True,
        type=bool,
        help=""
    )

    parser.add_argument(
        "-wx",
        default=3,
        type=int,
        help="past window size"
    )

    parser.add_argument(
        "-wy",
        default=1,
        type=int,
        help="future window size"
    )

    parser.add_argument(
        "-c",
        default=GradientBoostingClassifier,
        type=lambda clf: {"xg": GradientBoostingClassifier,
                          "tree": DecisionTreeClassifier,
                          "random_forest": RandomForestClassifier}[clf],
        help="Classifier (xg, tree or random forest)"
    )

    return parser.parse_args()


if __name__ == '__main__':
    df1 = load_data(rf"{prog_dir}/Data/Cleaned_Agg_Workouts_2023.csv")
    df2 = load_data(rf"{prog_dir}/Data/Cleaned_riderIllnesses.csv")
    df_merged = adding_labels.merge_selected_columns_from_dfs(df1, df2, ["disrupt"])
    args = parse_args()
    print(args)
    wx = args.wx
    wy = args.wy
    classifier = args.c
    print(f"chosen arguments are:"
          f"window x: {wx}"
          f"window y: {wy}"
          f"classifier: {classifier}")
    time_series_df = data_formatting_for_model.create_data_frame_for_model(
        df_merged,
        window_size_x=wx,
        window_size_y=wy,
        target_label_column_name="disrupt")

    for wx in [14]:
        for wy in [1, 2, 3, 7]:
            t = time.time()
            print(f"creating time series: x:{wx}, y:{wy}")
            time_series_df = data_formatting_for_model.create_data_frame_for_model(
                df_merged,
                window_size_x=wx,
                window_size_y=wy,
                target_label_column_name="disrupt")
            time_series_df.to_csv(rf"{prog_dir}/Data/TimeSeries_x{wx}_y{wy}.csv", index=False)
            print(f"created x:{wx}, y:{wy} in time {time.time() - t}")
#
#     # dtc = create_model(time_series_df, classifier=classifier)
# def plot_for_all_time_interval():
#     width = 0.14
#     fig, ax = plt.subplots(figsize=(18, 10))
#
#     def add_labels_on_bars(x,y):
#         y_text = [f"{val:.1%}" for val in y]
#         y_place = [val*0.5 for val in y]
#         for i in range(len(x)):
#             ax.text(x[i], y_place[i], y_text[i], ha='center', rotation='vertical', fontsize=6)
#
#     rel_cols = list(filter(lambda col: col[:10] == "Percentage", aggregated_df.columns))
#     rel_cols.sort()
#     x_locations = []
#     last_left_pos = - width * ((len(rel_cols) - 1) / 2)
#     for i, col in enumerate(rel_cols):
#         x_locations.append([x + last_left_pos for x in range(len(aggregated_df))])
#         last_left_pos += width
#
#         ax.bar(x_locations[i], list(aggregated_df[col]), width, label=col[10:])
#         add_labels_on_bars(x_locations[i], list(aggregated_df[col]))
#
#     labels = [x[:4] for x in list(aggregated_df.index)]
#     labels = [f"{str(x)[2:]}-{str(int(x) + interval_size)[2:]}" for x in labels]
#
#     ax.set_xticks(range(len(aggregated_df)), labels=labels, fontsize=8)
#     plt.xlabel('years', fontsize=10)
#     plt.ylabel('CARG%', fontsize=10)
#     plt.title(title)
#     plt.grid(color = 'green', linestyle = '--', linewidth = 0.1)
#     ax.legend()
#     plt.show()