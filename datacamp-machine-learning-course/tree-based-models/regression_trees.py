## Output is a real value. We want to predict a value
from sklearn.tree import DecistionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_suqred_error as MSE
from sklearn.model_selection import cross_val_score
import pandas as pd

pd.read_csv("auto.csv")

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.2,
                                                    stratify=y,
                                                    random_state=1)

## each leaf contains 10% of the data
dt = DecisionTreeRegressor(max_depth=2, min_sample_leaf=0.1, random_state=1)
# Instantiate dt_entropy, set 'entropy' as the information criterion
#dt_entropy = DecisionTreeClassifier(max_depth=8, criterion='entropy', random_state=1)
dt.fit(X_train,y_train)

y_pred = dt.predict(X_test)

## Compute test set MSE
mse_dt = MSE(y_test, y_pred)

## Compute test set RMSE
rmse_dt = mse_dt**(1/2)

print(rmse_dt)
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


# 
MSE_CV_scores = cross_val_score(dt, X_train, y_train, cv=10, socring='neg_mean_suqred_error', n_jobs=-1)

# Compute the 10-folds CV RMSE
RMSE_CV = (MSE_CV_scores.mean())**(1/2)

# Print RMSE_CV
print('CV RMSE: {:.2f}'.format(RMSE_CV))

dt.fit(X_train, y_train)

y_predict_train = dt.predict(X_train)

y_predict_test = dt.predict(X_test)

mse_train = MSE(y_train, y_predict_train)

mse_test = MSE(y_predict_test, y_test);
