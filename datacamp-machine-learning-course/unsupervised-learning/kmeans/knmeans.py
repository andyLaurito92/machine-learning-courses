from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd

import matplotlib.pyplot as plt


data = load_iris() # Returns a dictionary See more in https://scikit-learn.org/stable/modules/generated/sklearn.utils.Bunch.html#sklearn.utils.Bunch
samples = data.get('data') # Get the N x 4 matrix
labels = data.get('target')
len_samples = len(samples)

X_train = samples[1:int(len_samples * 0.8), :]
X_test = samples[len_samples - int(len_samples * 0.8):len_samples, :]
y_train = labels[0:int(len_samples * 0.8)]
y_test = labels[len_samples - int(len_samples * 0.8):len_samples]

#print(samples)
model = KMeans(n_clusters=3)

xs = X_train[:, 0]
ys = X_train[:, 2]
plt.scatter(xs, ys)
plt.show()

model.fit(X_train)
labels = model.predict(X_test)
# The above lines can be replaced by labels = model.fit_predict(samples)

print(labels)
print("Inertia ", model.inertia_) # Measure distace from each sample to centroid. = how spread
# out the cluster are (the lower, the better)

# Make a scatter plot of centroids_x and centroids_y
# Assign the cluster centers: centroids
centroids = model.cluster_centers_
print(centroids)

plt.scatter(centroids[:, 0], centroids[:, 1], marker='D', s=50)

xs = X_test[:, 0]
ys = X_test[:, 2]
plt.scatter(xs, ys, c=labels)
plt.show()


# Create a dataframe using pandas
species = data.get('target_names')
print(species)
df = pd.DataFrame({'labels': labels, 'species': y_train})
print(df)
ct = pd.crosstab(df['labels'], df['species'])
print(ct) 

