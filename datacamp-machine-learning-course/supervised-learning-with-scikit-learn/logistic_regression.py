from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, roc_auc_score

logreg = LogisticRegression() # threshold = 0.5 by default
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)

# 2 dimensional array
y_pred_probs = logreg.predict_proba(X_test)[:, 1] ## Probability of elements belonging to class
print(y_pred_probs[0])

# ROC --> Recieve operativing caracteristic (?
false_positive_rate, tpr, thresholds = roc_curve(y_test ,y_pred_probs)

plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Logistic Regression ROC Curve')
plt.show()


# ROC AUC --> Calculate the area under the curve
# With this we can evaluate the performance of our model
print(roc_auc_score(y_test, y_pred_probs)) 
