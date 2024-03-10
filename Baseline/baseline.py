from Preprocessing import data_manipulation, data_formatting_for_model, agg_workout_pp, adding_labels

import pandas as pd
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation


def evaluate_model(classifier: DecisionTreeClassifier, X_test: pd.DataFrame, y_test: pd.DataFrame):
    y_pred = classifier.predict(X_test)

    print("Accuracy:", metrics.accuracy_score(y_test, y_pred))


def create_model(df: pd.DataFrame):
    feature_cols = df.columns[:-1]
    label_col = [df.columns[-1]]
    X = df[feature_cols]
    y = df[label_col[0]]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                                        random_state=1)  # 70% training and 30% test

    clf = DecisionTreeClassifier()

    # Train Decision Tree Classifier
    clf = clf.fit(X_train, y_train)
    evaluate_model(clf, X_test, y_test)


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
    create_model(time_series_df)

