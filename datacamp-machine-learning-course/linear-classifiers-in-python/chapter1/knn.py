import sklearn.datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

newsgroups = sklearn.datasets.fetch_20newsgroups_vectorized()
knn = KNeighborsClassifier(n_neighbors=1)

X, y = newsgroups.data, newsgroups.target
X_train, X_test, y_train, y_test = train_test_split(X,y)

print(X.shape)
print(y.shape)

# Need to split train, cross validation and test train!
knn.fit(X_train,y_train)

print(knn.score(X_test, y_test))
