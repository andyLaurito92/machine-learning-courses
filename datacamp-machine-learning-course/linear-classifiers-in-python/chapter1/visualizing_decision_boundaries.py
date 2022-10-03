import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_wine

# You need to have scikit-learn version 1.1.1 to make this work
# It's already installed in the conda environment, but emacs is
# not using it. Starting point to read this: https://emacs.stackexchange.com/questions/20092/using-conda-environments-in-emacs
from sklearn.inspection import DecisionBoundaryDisplay

X, y= load_wine(return_X_y=True)

# Define the classifiers
classifiers = [LogisticRegression(), LinearSVC(), SVC(), KNeighborsClassifier()]

# Fit the classifiers
for c in classifiers: 
    c.fit(X, y)

# Plot the classifiers
plot_4_classifiers(X, y, classifiers)
plt.show()


def plot_4_classifiers(X, y, classifiers):
    titles = (
        "Logistic Regression",
        "Linear SVC",
        "SVC",
        "KNeighborsClassifier"
    )

    # Set-up 2x2 grid for plotting.
    fig, sub = plt.subplots(2, 2)
    plt.subplots_adjust(wspace=0.4, hspace=0.4)

    for clf, title, ax in zip(classifiers, titles, sub.flatten()):
        disp = DecisionBoundaryDisplay.from_estimator(
            clf,
            X,
            response_method="predict",
            cmap=plt.cm.coolwarm,
            alpha=0.8,
            ax=ax,
            xlabel=iris.feature_names[0],
            ylabel=iris.feature_names[1],
        )
        ax.scatter(X0, X1, c=y, cmap=plt.cm.coolwarm, s=20, edgecolors="k")
        ax.set_xticks(())
        ax.set_yticks(())
        ax.set_title(title)
    plt.show()
