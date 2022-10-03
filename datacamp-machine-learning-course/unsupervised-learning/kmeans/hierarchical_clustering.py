# Data here is taken from voting countries at the Eurovision song contest

import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster

# Linkage defines at which distance we should stop merging cluster. Complete means
# that we need to keep up merging clusters up to the end
## In complete linkage, the distance between clusters is the distance between the furthest points of the clusters. In single linkage, the distance between clusters is the distance between the closest points of the clusters.
mergings = linkage(samples, method='complete') # This method performs the dendrogram
labels = fcluster(mergings, 15, criterion='distance')
print(labels) # numpy array
# fcluster extracts labels from a cluster of labels

dendrogram(merginsg, labels=country_names, leaf_rotation = 90, leaf_font_size=6)
plt.show()

# Aligning cluster labels with country names
pairs = pd.DataFrame({'labels': labels, 'countries': country_names})
print(pairs.sort.values('labels'))
