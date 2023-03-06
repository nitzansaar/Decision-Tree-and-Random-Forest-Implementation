import sklearn.datasets
from collections import Counter
import random
from sklearn import tree

## first, install scikit-learn

## This code loads the iris dataset. Change it to load the breast cancer
# dataset instead.

iris = sklearn.datasets.load_iris()


## iris has two member variables we're going to use right now - data and target.
## data are the attributes (or independent variables) and target are the classifications.
## all the sklearn datasets have this feature.

## Now we'll implement our first learning algorithm: ZeroR. ZeroR is a really
## boring learning algorithm. It just chooses the most common value from the
## target. It will be a good baseline.

## assume we're passing in the dataset. You do the rest.
def ZeroR(data):
    pass


## Slightly more interesting is RandR. RandR selects from among the target values
# according to their frequency. So if we have:
# ['cat','cat','cat','dog'] as our targets, we should choose 'cat' 75% of the
# time, and 'dog' 25% of the time. You make this.

def RandR(data):
    pass

## Now let's see how to use sklearn to build a decision tree.
## https://scikit-learn.org/stable/modules/tree.html#decision-trees

iris_tree = tree.DecisionTreeClassifier()
iris_tree.fit(iris.data, iris.target)

## Now that we've built a tree, we can use it to predict a value.
## Let's test the first element in the dataset.
predicted_val = iris_tree.predict([iris.data[0]])
print("Predicted value: %d. True value: %d." % (predicted_val[0], iris.target[0]))

## But of course the tree gets it right - we're testing on the same data we trained
## on! We need to test on different data that we train with to see if it actually works.

## Let's start by splitting the data into training and test sets.

## the iris data is ordered. We'd like to shuffle it, but we want to keep the
## data and targets together. Let's do this by creating tuples with zip.

pairs = list(zip(iris.data, iris.target))

# then we'll shuffle the list.
random.shuffle(pairs)

# Now let's take the first 80% as training, and the last 20% as test.
threshold = int(0.8 * len(pairs))

training_set = pairs[:threshold]
test_set = pairs[threshold:]

## Now let's train our classifier again, just with the training set.

iris_tree = tree.DecisionTreeClassifier()
iris_tree.fit([item[0] for item in training_set],
              [item[1] for item in training_set])

## And then let's test all of the items in our test set.
predicted_vals = iris_tree.predict([item[0] for item in test_set])

## we want to iterate through predicted_vals and the test_set and compare them.
## let's do this with zip.
correct = 0
for thing in zip(predicted_vals, [item[1] for item in test_set]):
    print("Predicted value: %d Actual value: %d" % (thing[0], thing[1]))
    if thing[0] == thing[1]:
        correct = correct + 1

print("Total correct: %d Percent correct: %f" % (correct, correct / len(predicted_vals)))

## But wait - we still might not have an accurate measure of performance.
## We would like to test our tree with as much different data as possible
## to get a precise estimate of accuracy. We do this with cross-validation.

## Here we'll do 5-fold cross-validation, first by hand, then with sklearn.
## The idea is that we break our dataset into 5 buckets, or folds.
## We then train our tree five times. Each time we use four of the folds as
## a training set, and the fifth as a test set. This way, we know that all data
## has been used for both training and testing, and we reduce the risk of
## accidentally selecting a bad test set.

pairs = list(zip(iris.data, iris.target))
random.shuffle(pairs)

folds = 5
fold_size = int(len(pairs) / folds)
accuracies = [0, 0, 0, 0, 0]

for i in range(folds):
    lower_bound = fold_size * i
    upper_bound = fold_size * (i + 1)

### you figure out the rest!

## We can also ask sklearn to do cross-validation for us.
from sklearn.model_selection import cross_val_score

iris_tree = tree.DecisionTreeClassifier()
accuracies = cross_val_score(iris_tree, iris.data, iris.target, cv=5)

print(accuracies)

## As we know, Decision Trees are vulnerable to overfitting. We can
## overcome this through the use of a random forest, which is a collection
## of trees that we sample from.

from sklearn.ensemble import RandomForestClassifier

iris_forest = RandomForestClassifier(n_estimators=5)
iris_forest.fit(iris.data, iris.target)
print(iris_forest.predict(iris.data))

## We can combine this with the five-fold cross-validation above.

iris_forest = RandomForestClassifier(n_estimators=5)
accuracies = cross_val_score(iris_forest, iris.data, iris.target, cv=5)
print(accuracies)


