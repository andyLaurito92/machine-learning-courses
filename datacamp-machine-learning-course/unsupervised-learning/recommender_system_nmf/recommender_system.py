# apply NMF to the word-frequency array over the articles
# nmf feature values describe the topics
# similar documents have similar nmf feature values

from sklearn.preprocessing import normalize
from sklearn.decomposition import NMF
import pandas as pd

nmf = NMF(n_components=6)
nmf_features = nmf.fit_transform(articles)

# How to compare articels?
# Use the cosine similarity
# higher values means more similar :)

norm_features = normalize(nmf_features)

# if has index 23
current_article = norm_features[23, :]
similarities = norm_features.dot(current_articles)
print(similarities)


df = pd.DataFrame(norm_features, index=titles)
current_article = df.loc['Dog bites man']
similarities = df.dot(current_article)
print(similarities.nlargest())
