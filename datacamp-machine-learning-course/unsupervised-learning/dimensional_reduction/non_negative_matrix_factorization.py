# This is a reduction technique. The models produced by NMF are interpretable, as in contrast to PCA
#  To apply NMF, all sample features must be non-negative >= 0
# Read more about NMF here: https://en.wikipedia.org/wiki/Non-negative_matrix_factorization

from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer 
import pandas as pd

articles = pd.read_csv('/Users/andylaurito/Desktop/machine-learning/datacamp-machine-learning-course/unsupervised-learning/wikipedia_mining_text/wikipedia_utf8_filtered_20pageviews.csv')
documents = ['the cat', 'the cat is under the table', 'the dog loves to bark', 'dog barks cat']

# tf = frequency of word in the document
# idf = reduces influence of frequent words, like the

tfidf = TfidfVectorizer()
samples = tfidf.fit_transform(samples)

model = NMF(n_components=2) # n_components is a mandatory argument :)
model.fit(samples)

nmf_features = model.transform(samples)

#print(samples)
print(model.components_) # In here you have the topics of a document, or the parts of an image :)


## NMF used in articles gives you the most representatives word of the articles...
## It gives you the topics that characterizes the articles!! :) The idea behind this
## is that it gives you the patterns that are more representative to a document


## Show an image :)
## from matplot import pyplot as plt
# plt.imshow(bitmap, cmap='gray', interpolation='nearest')
# plt.show
