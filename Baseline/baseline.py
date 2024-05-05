from MiscUtils import load_data
from Preprocessing import data_manipulation, data_formatting_for_model, agg_workout_pp, adding_labels

import pandas as pd
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
from sklearn.metrics import classification_report


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


def create_model(df: pd.DataFrame, classifier=DecisionTreeClassifier):
    feature_cols, label_cols = filter_relevant_cols_for_model(df)
    # feature_cols = df.columns[:-1]
    # label_col = [df.columns[-1]]
    df = df.apply(pd.to_numeric, errors='coerce')
    X = df[feature_cols]
    y = df[label_cols]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4,
                                                        random_state=1)  # 90% training and 10% test

    clf = DecisionTreeClassifier(max_depth=3, min_samples_split=2, min_samples_leaf=1, max_features=None)

    # Train Decision Tree Classifier
    clf = clf.fit(X_train, y_train)
    evaluate_and_visualize_model(clf, X_test, y_test)
    return clf


if __name__ == '__main__':
    df1 = load_data(r"../Data/Cleaned_Agg_Workouts_2023.csv")
    df2 = load_data(r"../Data/Cleaned_riderIllnesses.csv")
    df_merged = adding_labels.merge_selected_columns_from_dfs(df1, df2, ["disrupt"])
    time_series_df = data_formatting_for_model.create_data_frame_for_model(
        df_merged,
        window_size_x=4,
        window_size_y=1,
        target_label_column_name="disrupt")
    dtc = create_model(time_series_df)
