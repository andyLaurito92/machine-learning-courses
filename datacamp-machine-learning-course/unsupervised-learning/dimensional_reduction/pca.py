from sklearn.decomposition import PCA
model = PCA()
model.fit(samples)
transformed = model.transform(samples)
# Columns of transformed are the "PCA features", row gives PCA feature values of corresponding sample

# NOTE: Resulting PCA features are not linearly correlated ("decorrelation")
# Pearson correlation measures linear correlation
print(transformed)

# Principal components attribute
print(model.components_) # each row defines mean of sample


## Decorrelating measurements

# Import PCA

# Create PCA instance: model
model = PCA()

# Apply the fit_transform method of model to grains: pca_features
pca_features = model.fit_transform(grains)

# Assign 0th column of pca_features: xs
xs = pca_features[:,0]

# Assign 1st column of pca_features: ys
ys = pca_features[:,1]

# Scatter plot xs vs ys
plt.scatter(xs, ys)
plt.axis('equal')
plt.show()

# Intrinsic dimension
# Number of features needed to approximate the dataset
# It tell us the most compact representation of the samples
# PCA help us detect the intrisc dimension. These are the number of
# PCA features with significant variance :)

## Intrinsic dimension is number of PCA features with significant variance :)
