# Using cross_tabulation tables with pandas ;)
import matplotlib.pyplot as plt
import pandas as pd


data = load_iris() # Returns a dictionary See more in https://scikit-learn.org/stable/modules/generated/sklearn.utils.Bunch.html#sklearn.utils.Bunch
species = data.get('target_names')
print(species)
df = pd.DataFrame({'labels': labels, 'species': species})
print(df)
