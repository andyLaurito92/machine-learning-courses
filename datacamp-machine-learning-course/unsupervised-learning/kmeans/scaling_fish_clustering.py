## Note that Normalizer() is different to StandardScaler(), which you used in the previous exercise. While StandardScaler() standardizes features (such as the features of the fish data from the previous exercise) by removing the mean and scaling to unit variance, Normalizer() rescales each sample - here, each company's stock price - independently of the other.


from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import Normalizer

import pandas as pd

data = pd.read_csv('fishdata.csv')
data = data[0:85]
scaler = StandardScaler()
kmeans = KMeans(n_clusters=4)
imp_cat = SimpleImputer(strategy="most_frequent")
pipeline = make_pipeline(imp_cat, scaler, kmeans)

species = ['Bream', 'Bream', 'Bream', 'Bream', 'Bream', 'Bream', 'Bream', 'Bream', 'Bream', 'Bream', 'Bream', 'Bream', 'Bream', 'Bream', 'Bream', 'Bream', 'Bream', 'Bream', 'Bream', 'Bream', 'Bream', 'Bream', 'Bream', 'Bream', 'Bream', 'Bream', 'Bream', 'Bream', 'Bream', 'Bream', 'Bream', 'Bream', 'Bream', 'Bream', 'Roach', 'Roach', 'Roach', 'Roach', 'Roach', 'Roach', 'Roach', 'Roach', 'Roach', 'Roach', 'Roach', 'Roach', 'Roach', 'Roach', 'Roach', 'Roach', 'Roach', 'Roach', 'Roach', 'Roach', 'Smelt', 'Smelt', 'Smelt', 'Smelt', 'Smelt', 'Smelt', 'Smelt', 'Smelt', 'Smelt', 'Smelt', 'Smelt', 'Smelt', 'Smelt', 'Smelt', 'Pike', 'Pike', 'Pike', 'Pike', 'Pike', 'Pike', 'Pike', 'Pike', 'Pike', 'Pike', 'Pike', 'Pike', 'Pike', 'Pike', 'Pike', 'Pike', 'Pike']

pipeline.fit(data)
labels = pipeline.predict(data)
df = pd.DataFrame({'labels':labels, 'species':species})
ct = pd.crosstab(df['labels'], df['species'])

print(ct)
