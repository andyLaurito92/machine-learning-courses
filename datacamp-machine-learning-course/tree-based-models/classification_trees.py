from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd

pd.read_csv("auto.csv")

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.2,
                                                    stratify=y,
                                                    random_state=1)

dt = DecisionTreeClassifier(max_depth=2, random_state=1)
# Instantiate dt_entropy, set 'entropy' as the information criterion
#dt_entropy = DecisionTreeClassifier(max_depth=8, criterion='entropy', random_state=1)
dt.fit(X_train,y_train)

y_pred = dt.predict(X_test)

accuracy_score(y_test, y_pred)

## Trees generate decision regions that look like a square. This happens
## because at each node, 1 feature is used to define which path to take


## How to build the tree?

# At each node, we have a feature f and a splitting point sp, and we try to answer:

## f < sp?
## if yes, left
## if no, right

## How do we know which feature to select in each node and how to define the splitting point?

## We maximize information gain
## IG(f, sp) = I(parent) - (N_left/N * I(left) + N_right/N * I(right))
## where I = impurity

## What is impurity and how do we define it?
## We can measure it by using:
## Gini index --> faster to compute
## entropy

## If IG(node) = 0, declare the node of leaf (This is when max_dpeth is not defined)

