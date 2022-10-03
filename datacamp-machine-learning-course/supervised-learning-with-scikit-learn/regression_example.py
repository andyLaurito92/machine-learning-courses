from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


X = sales_df['radio'].values

# Create y from the sales column's values
y = sales_df['sales'].values

# Reshape X
X = np.reshape(-1, 1)

# Check the shape of the features and targets
print(X.shape, y.shape)

reg = LinearRegression()
reg.fit(X_bmi, y)
predictions = reg.predict(X_bmi)

# mean_squared_error(y_test, y_pred, squared=False)

# pandas.drop('[column_name]', axis=1) # axis = 1 ==> drop column

# Compute R-squared
r_squared = reg.score(X_test, y_test)

# Compute RMSE
rmse = mean_squared_error(y_test, y_pred, squared=True)

# Print the metrics
print("R^2: {}".format(r_squared))
print("RMSE: {}".format(rmse))

plt.scatter(X_bmi, y)
plt.plot(X_bmi, predictions)
plt.ylabel("Blood Glucose (mg/dl)")
plt.xlabel("Body Mass Index")
plt.show()
