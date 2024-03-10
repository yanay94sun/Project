from Preprocessing import data_manipulation, data_formatting_for_model, agg_workout_pp

import pandas as pd
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation


def evaluate_model(classifier: DecisionTreeClassifier, X_test: pd.DataFrame, y_test: pd.DataFrame):
    y_pred = classifier.predict(X_test)

    print("Accuracy:", metrics.accuracy_score(y_test, y_pred))


def create_model(df: pd.DataFrame):
    feature_cols = []
    label_col = []
    X = df[feature_cols]
    y = df[label_col]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                                        random_state=1)  # 70% training and 30% test

    clf = DecisionTreeClassifier()

    # Train Decision Tree Classifier
    clf = clf.fit(X_train, y_train)
    evaluate_model(clf, X_test, y_test)


if __name__ == '__main__':
    data_formatting_for_model.create_data_frame_for_model(window_size_x=3, window_size_y=1, )
    create_model()

