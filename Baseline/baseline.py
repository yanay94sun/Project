from Preprocessing import data_manipulation, data_formatting_for_model, agg_workout_pp, adding_labels

import pandas as pd
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation


def evaluate_and_visualize_model(classifier: DecisionTreeClassifier, X_test: pd.DataFrame, y_test: pd.DataFrame):
    y_pred = classifier.predict(X_test)

    print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
    import matplotlib.pyplot as plt
    from sklearn import tree

    # Assuming clf is your trained DecisionTreeClassifier object
    plt.figure(figsize=(12, 8))
    tree.plot_tree(classifier, filled=True, feature_names=X_test.columns)
    plt.show()



def filter_relevant_cols_for_model(df: pd.DataFrame):
    feature_cols = df.columns[:-1]  # removing the label column
    remove_keys = ["date", "day", "year", "month", "_id"]
    feature_cols = list(filter(lambda col: not any([key in col for key in remove_keys]), feature_cols))
    return feature_cols, [df.columns[-1]]


def create_model(df: pd.DataFrame):
    feature_cols, label_cols = filter_relevant_cols_for_model(df)
    # feature_cols = df.columns[:-1]
    # label_col = [df.columns[-1]]
    df = df.apply(pd.to_numeric, errors='coerce')
    X = df[feature_cols]
    y = df[label_cols]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                                        random_state=1)  # 70% training and 30% test

    clf = DecisionTreeClassifier(max_depth=None, min_samples_split=2, min_samples_leaf=1, max_features=None)

    # Train Decision Tree Classifier
    clf = clf.fit(X_train, y_train)
    evaluate_and_visualize_model(clf, X_test, y_test)
    return clf


if __name__ == '__main__':
    df1 = data_manipulation.prepare_data_for_decision_tree(r"C:\Users\t-asafstern\Desktop\Studies\StudiesProjext\Confidential\Confidential\Agg_Workouts_2023.csv")
    df2 = data_manipulation.prepare_data_for_decision_tree(r"C:\Users\t-asafstern\Desktop\Studies\StudiesProjext\Confidential\Confidential\riderIllnesses.csv")
    df_merged = adding_labels.merge_selected_columns_from_dfs(df1, df2, ["disrupt"])
    # df_merged['date'] = f"{df_merged['year']}-{df_merged['month']}-{df_merged['day']}"
    df_merged['date'] = pd.to_datetime(df_merged[['year', 'month', 'day']])
    time_series_df = data_formatting_for_model.create_data_frame_for_model(
        df_merged,
        window_size_x=3,
        window_size_y=1,
        target_label_column_name="disrupt")
    time_series_df = time_series_df.dropna()
    dtc = create_model(time_series_df)


