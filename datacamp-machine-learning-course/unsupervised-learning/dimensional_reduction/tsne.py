# t-SNE = "t-distributed stochastic neighbor embedding"
# Maps samples of high dimensional space to 2d or 3d space ( Similar to PCA? )
# With this, we can visualize them :)
# Maps approximately preserves nearness of samples

# samples = load_flowers_data

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
model = TSNE(learning_rate = 100)
transformed = model.fit_transform(samples)
xs = transformed[:, 0]
ys = transformed[:, 1]

plt.scatter(xs, ys, c=species)

# With this code you can annotate the points in the plot
## # Annotate the points
#for x, y, company in zip(xs, ys, companies):
#    plt.annotate(company, (x, y), fontsize=5, alpha=0.75)

plt.show()
