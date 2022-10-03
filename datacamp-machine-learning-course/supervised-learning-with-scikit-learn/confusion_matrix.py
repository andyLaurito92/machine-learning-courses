from sklearn.metrics import classification_report, confusion_matrix

knn = KNeighborsClassifier(n_neighbors=7)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

print(confusion_matrix(y_test, y_pred)) # Calculates the confusion amtrix!
print(classification_report(y_test, y_pred))

#hyperparameters --> alpha/n_neighbors

## How to chose hyperparameters?. Hyperparamter tunning
# 1. Try lots of different hyperparameter values
# 2. Fit all of them separately
# 3. See how well they perform
# 4. Choose the best performing values


# For the above, use cross validation
