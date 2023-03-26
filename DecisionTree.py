import pandas as pd
from collections import Counter
from math import log2 as log

from sklearn.model_selection import KFold
from sklearn_intro import ZeroR


### take as input a Pandas series and return the entropy of this series
def entropy(values) :
    c = Counter(values)
    ent = 0
    distinct = set(values)
    for item in distinct :
        frequency = c[item] / len(values)
        ent += -1 * frequency * log(frequency)
    return ent

## takes as input two pandas series - one representing the variable
## to be tested, and the other the corresponding classifications.
## return the remainder - the weighted average of entropy.
def remainder(variables, classifications) :
    vars_list = set(variables)
    unique_vars = list(variables)
    class_list = list(classifications)
    rem = 0.0
    for var in vars_list:
        ent = entropy([item[1] for item in zip(unique_vars, class_list) if item[0] == var])
        rem += (unique_vars.count(var) / len(unique_vars)) * ent
        # print(f"{var} {rem} {ent}")
    return rem

def select_attribute(df, classifications) :
    min_remainder = 100
    selected_column = ''

    for col in df:
        remainder_val = remainder(df[col], classifications)
        print(f"column: {col}, remainder: {remainder_val}")
        if remainder_val < min_remainder:
            min_remainder = remainder_val
            selected_column = df[col].name

    return selected_column

class Node :
    def __init__(self, classification=None,attribute=None):
        self.classification = classification
        self.attribute = attribute
        self.children = {}

    def isLeaf(self):
        return len(self.children) == 0

def make_tree(dataframe, classifications, attributes):
    if entropy(classifications) == 0 : #Base case 1
        return Node(dataframe, classifications) # return leaf
    if dataframe.empty or dataframe.columns.empty : #Base case 2
        data = ZeroR(dataframe)
        return Node(data, classifications)
    selected_attribute = select_attribute(dataframe, classifications)
    node = Node(attribute=selected_attribute) #non leaf node
    possible_values = attributes[selected_attribute]
    for value in possible_values:
        sub_dataframe = dataframe[dataframe[selected_attribute] == value].drop(columns=[selected_attribute]) #split data and remove attribute
        sub_classifications = classifications[dataframe[selected_attribute] == value]
        node.children[value] = make_tree(sub_dataframe, sub_classifications, attributes)
    return node

def read_file(file_path):
    df = pd.read_csv(file_path)
    dataframe = df.iloc[:, :-1]
    classifications = df.iloc[:, -1]
    return dataframe, classifications

def classify(tree, to_classify) :
    if tree.isLeaf():
        return tree.classification
    attribute_value = to_classify[tree.attribute]
    if attribute_value in tree.children:
        return classify(tree.children[attribute_value], to_classify)
    else:
        return ZeroR(tree)
def create_attributes_dict(dataframe):
    attributes = {}
    for col in dataframe.columns:
        attributes[col] = dataframe[col].drop_duplicates().tolist()
    return attributes











