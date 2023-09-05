# Decision Tree and Random Forest Implementation

This repository contains Python code for implementing decision trees, random forests, and performing classification using them. The code includes implementations of decision tree algorithms, entropy calculations, information gain, attribute selection, and classification using decision trees and random forests.

## Contents

1. [Entropy Calculation](#entropy-calculation)
2. [Attribute Selection](#attribute-selection)
3. [Decision Tree Implementation](#decision-tree-implementation)
4. [Random Forest Implementation](#random-forest-implementation)
5. [Testing](#testing)
6. [What I Learned](#what-i-learned)

---

## Entropy Calculation

The `DecisionTree` module includes functions to calculate entropy, a measure of impurity in a dataset. The `entropy` function takes a Pandas series as input and calculates the entropy of the data based on the class distribution.

## Attribute Selection

The `DecisionTree` module provides functions to compute the remainder of a dataset after splitting it based on different attributes. The `remainder` function calculates the weighted average of entropy for different attribute values. The `select_attribute` function selects the attribute that results in the lowest remainder.

## Decision Tree Implementation

The `DecisionTree` module contains functions to build a decision tree using the provided dataset. The `make_tree` function constructs a decision tree recursively by selecting attributes with the lowest remainder. The `Node` class represents a node in the decision tree, containing attribute and classification information.

## Random Forest Implementation

The `RandomForestClassifier` from the `sklearn.ensemble` module is used to create a random forest of decision trees. The random forest combines multiple decision trees to make predictions. The example in the code uses the breast cancer dataset for classification using a random forest.

## Testing

The code includes unit tests for the `entropy`, `remainder`, `select_attribute`, and `make_tree` functions. The test cases cover different scenarios and datasets to ensure the correctness of the implemented functions.

---

## What I Learned

- How to calculate entropy and use it to measure the impurity of a dataset.
- How attribute selection and information gain play a crucial role in constructing decision trees.
- The process of recursively building a decision tree and understanding the structure of nodes.
- Implementing a random forest classifier and leveraging the power of ensemble methods.
- Performing cross-validation to evaluate model performance and handle overfitting.
- Using unit tests to verify the correctness of implemented functions.

---

## How to Use

1. Clone or download the repository.
2. Install the required dependencies by running `pip install -r requirements.txt` (make sure you have Python and pip installed).
3. Run the desired Python scripts or use the code as needed.