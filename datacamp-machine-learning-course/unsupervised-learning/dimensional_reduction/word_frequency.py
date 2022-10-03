# In word frequency arrays, rows represent documents and columns are the words. Therefore, (i, j) specifies how many times word_j is repetead in document_i

#from scipy.sparse import csr_matrix # sparse arrays :)

# scikit-learn PCA doesn't support csr_matrix. We need to use TrucatedSVD instead!

# TfidfVectorizer  transforms a list of documents into a word frequency array, which it outputs as a csr_matrix.
from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.decomposition import TruncatedSVD

documents = ['cats say meow', 'dogs say woof', 'dogs chase cats']
# model = TruncatedSVD(n_components=3)
# model.fit(documents)
# transformed=model.transform(documents)

# Create a TfidfVectorizer: tfidf
tfidf = TfidfVectorizer() 

# Apply fit_transform to document: csr_mat
csr_mat = tfidf.fit_transform(documents)

print(documents)
# Print result of toarray() method
print(csr_mat.toarray())

# Get the words: words
words = tfidf.get_feature_names()

# Print words
print(words)
