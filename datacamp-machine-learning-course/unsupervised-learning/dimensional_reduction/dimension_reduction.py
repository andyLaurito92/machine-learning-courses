# Reduce the dimension features to N
# You need to know to how many dimensions you want to reduce the data. For this, you can use the intrinsic dimension. In order to get this, take a look at ploting_pca_features.py :)

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

pca = PCA(n_components=2)
pca.fit(samples)
transformed = pca.transform(samples)
print(transformed.shape)

xs = transformed[:, 0]
ys = transformed[:, 1]
plt.scatter(xs, ys, c=species)
plt.show()
