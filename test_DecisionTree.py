from unittest import TestCase
import pandas as pd

import DecisionTree


class Test(TestCase):
    def test_entropy(self):
        ## e should be 1
        test_case1 = pd.Series([1, 2, 1, 2, 1, 2, 1, 2])
        print(DecisionTree.entropy((test_case1)))
        ## e should be 0
        test_case2 = pd.Series([1, 1, 1, 1, 1, 1, 1, 1])
        print(DecisionTree.entropy((test_case2)))
        ## e should be 2
        test_case3 = pd.Series([1, 2, 3, 4, 1, 2, 3, 4])
        print(DecisionTree.entropy((test_case3)))


class Test(TestCase):
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
        test_classes2 = [0, 0, 0, 0, 1, 1, 1, 1]
        ## remainder should be 1
        print(DecisionTree.remainder(test_variables, test_classes2))
        ## remainder should be 0.69
        print(DecisionTree.remainder(tennis_variables, tennis_classes))
        data = pd.read_csv('tennis.csv')
        indep_vars = data['outlook']
        dep_vars = data['play']
        print(DecisionTree.remainder(indep_vars, dep_vars))


class Test(TestCase):
    def test_select_attribute(self):
        data = pd.read_csv('tennis.csv')
        indep_vars = data[data.columns[:-2]]
        dep_vars = data['play']
        print(DecisionTree.select_attribute(indep_vars, dep_vars))
        ## answer should be 'outlook'
