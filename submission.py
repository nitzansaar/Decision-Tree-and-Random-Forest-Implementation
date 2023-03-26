import random
from unittest import TestCase

import pandas as pd
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import DecisionTree
from sklearn_intro import ZeroR, RandR, breast_cancer
import NLTK_intro

# The NLTK code takes about a minute to run, you can comment it out to run the rest of the code quickly
class Test(TestCase):
    def test_entropy(self):
        ## e should be 1
        test_case1 = pd.Series([1, 2, 1, 2, 1, 2, 1, 2])
        # print(test_case1)
        print(DecisionTree.entropy((test_case1)))
        ## e should be 0
        test_case2 = pd.Series([1, 1, 1, 1, 1, 1, 1, 1])
        print(DecisionTree.entropy((test_case2)))
        ## e should be 2
        test_case3 = pd.Series([1, 2, 3, 4, 1, 2, 3, 4])
        print(DecisionTree.entropy((test_case3)))

    def test_remainder(self):
        tennis_variables = ['sunny', 'sunny', 'sunny', 'sunny', 'sunny',
                            'overcast', 'overcast', 'overcast', 'overcast',
                            'rainy', 'rainy', 'rainy', 'rainy', 'rainy']
        tennis_classes = ['yes', 'yes', 'no', 'no', 'no',
                          'yes', 'yes', 'yes', 'yes',
                          'yes', 'yes', 'yes', 'no', 'no']

        test_variables = [1, 2, 1, 2, 1, 2, 1, 2]
        test_classes1 = [0, 1, 0, 1, 0, 1, 0, 1]
        ## remainder should be 0
        print(DecisionTree.remainder(test_variables, test_classes1))
        assert DecisionTree.remainder(test_variables, test_classes1) == 0
        test_classes2 = [0, 0, 0, 0, 1, 1, 1, 1]
        ## remainder should be 1
        print(DecisionTree.remainder(test_variables, test_classes2))
        assert DecisionTree.remainder(test_variables, test_classes2) == 1
        ## remainder should be 0.69
        print(DecisionTree.remainder(tennis_variables, tennis_classes))
        data = pd.read_csv('tennis.csv')
        indep_vars = data['outlook']
        dep_vars = data['play']
        print(DecisionTree.remainder(indep_vars, dep_vars))

    def test_select_attribute(self):
        data = pd.read_csv('tennis.csv')
        indep_vars = data[data.columns[:-2]]
        print(indep_vars)
        dep_vars = data['play']
        print(dep_vars)
        print(DecisionTree.select_attribute(indep_vars, dep_vars))
        ## answer should be 'outlook'
        assert DecisionTree.select_attribute(indep_vars, dep_vars) == 'outlook'
    # Test tree and compare accuracy using five-fold cross-validation
    def test_make_tree(self):
        files = ["restaurant.csv", "tennis.csv", "breast-cancer.data"]
        for file in files:
            print(file)
            dataframe = pd.read_csv(file)
            for column in dataframe.columns:
                if dataframe[column].dtype == 'object':
                    label_encoder = LabelEncoder()
                    dataframe[column] = label_encoder.fit_transform(dataframe[column])
            feature_cols = dataframe.columns[:-1]
            target_col = dataframe.columns[-1]
            features = dataframe[feature_cols]
            target = dataframe[target_col]
            decision_tree = tree.DecisionTreeClassifier()
            random_forest = RandomForestClassifier()
            folds = 5
            for fold in range(folds):
                X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=1 / folds)
                decision_tree.fit(X_train, y_train)
                tree_predictions = decision_tree.predict(X_test)
                random_forest.fit(X_train, y_train)
                forest_predictions = random_forest.predict(X_test)
                tree_correct = accuracy_score(y_test, tree_predictions, normalize=False)
                tree_percent_correct = accuracy_score(y_test, tree_predictions) * 100
                forest_correct = accuracy_score(y_test, forest_predictions, normalize=False)
                forest_percent_correct = accuracy_score(y_test, forest_predictions) * 100
                print(f"Decision Tree: Total correct: {tree_correct} Percent correct: {tree_percent_correct:.2f}%")
                print(f"Random Forest: Total correct: {forest_correct} Percent correct: {forest_percent_correct:.2f}%")

    def test_classify(self):
        files = ["restaurant.csv", "tennis.csv", "breast-cancer.data"]
        for file in files:
            dataframe, classifications_series = DecisionTree.read_file(file)
            decision_tree = DecisionTree.make_tree(dataframe, classifications_series, DecisionTree.create_attributes_dict(dataframe))
            to_classify = dataframe.iloc[0]
            predicted_classification = DecisionTree.classify(decision_tree, to_classify)
            print(f"Predicted classification: {predicted_classification}\n")
