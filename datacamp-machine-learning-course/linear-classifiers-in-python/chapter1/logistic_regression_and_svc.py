from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
import sklearn.datasets


wine = sklearn.datasets.load_wine()

lr = LogisticRegression()
lr.fit(wine.data, wine.target)
#lr.predict(X_test)
lr.score(wine.data, wine.target)

lr.predict_proba(wine.data[:1])
