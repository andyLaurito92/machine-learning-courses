from sklearn.linear_model import Ridge ## This regularization is alpha * sum(a_{i} ^ 2)
from sklearn.linear_model import Lasso ## This regularization is alpha * sum(|a_{i}|)
import matplot.pyplot as plt

scores = []

for alpha in [0.1, 1.0, 10.0, 100.0, 1000.0]:
    ridge = Ridge(alpha=alpha) ## Can replace here for Lasso regression
    ridge.fit(X_train, y_train)
    y_pred = ridge.predict(X_test)
    scores.append(ridge.score(X_test, y_test))

print(scores)


## Lasso can be used for feature selection
X = diabetes_df.drop("glucose", axis=1).values
y = diabetes_df["glucose"].values
names = diabetes_df.drop("glucose", axis=1).columns
lasso = Lasso(alpha=0.1)
lasso_coef = lasso.fit(X, y).coef_
plt.bar(names, lasso_coef)
plt.xticks(rotation=45)
plt.show() ## This will show which features are important to determine the y predictor
