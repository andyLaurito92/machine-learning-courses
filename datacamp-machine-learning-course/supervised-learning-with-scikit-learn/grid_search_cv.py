from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
kf = KFold(n_splits=5, shuffle=True, random_state=42)

param_grid = {"alpha": np.arrange(0.0001, 1, 10), "solver": ["sag", "lsqr"]} # hyperparameters

ridge = Ridge()
#ridge_cv = GridSearchCv(ridge, param_grid, cv=kf)
ridge_cv = RandomizedSearchCV(ridge, param_grid, cv=kf, n_iter=2)
ridge_cv.fit(X_train, y_train)
print(ridge_cv.best_params_, ridge_cv.best_score_)

# Number of fits = number of hyperparamters * number of fold * number of values --> Doesn't scale!

#RnadomizedSearchCV


# Evalute on the test --> How well did my linear regression + regularization did?
test_score = ridge_cv.score(X_test, y_test)

# Create the parameter space
params = {"penalty": ["l1", "l2"],
         "tol": np.linspace(0.0001, 1.0, 50),
         "C": np.linspace(0.1, 1.0, 50),
         "class_weight": ["balanced", {0:0.8, 1:0.2}]}

# Instantiate the RandomizedSearchCV object
logreg_cv = RandomizedSearchCV(logreg, params, cv=kf)

# Fit the data to the model
logreg_cv.fit(X_train, y_train)

# Print the tuned parameters and score
print("Tuned Logistic Regression Parameters: {}".format(logreg_cv.best_params_))
print("Tuned Logistic Regression Best Accuracy Score: {}".format(logreg_cv.best_score_))
