# Encoding dummy variables
import pandas as pd
from sklearn.model_selection import cross_val_score, KFold
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

#scikit-learn requires numeric data, no missing values
# This means that if we have categorical features, such as color,
# scikit-learn will not accept these ones!. We need to convert these into numeric values :)

# Convert to binary features called dummy variables:
# 0 --> observation NOT from category
# 1 ---> observation FROM category

# Convert genre not numerical to dummy variables
music_df = pd.read_csv('my_csv.csv')
music_dummies = pd.get_dummies(music_df["genre"], drop_first=True) # Keep 9 of 10 features

# If the data frame has only 1 categorical feature, you can just pass the panda dataframe
music_dummies = pd.get_dummies(music_df, drop_first=True) # Keep 9 of 10 features
print(music_dummies.head()) 

# Convert back to genre
music_dummies = pd.concat([music_df, music_dummies], axis=1)
music_dummies = music_dummies.drop("genre", axis=1)

# Once we have our data as numerical values, we can go back to our linear regression models! :)


X = music_dummies.drop("popularity", axis=1).values
y = music_dummies["popularity"].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
kf = KFold(n_splits=5, shuffle=True, random_state=42)
linreg = LinearRegression()
linreg_cv = cross_val_score(linreg, X_train, y_train, cv=kf, scoring="neg_mean_squared_error")
print(np.sqrt(-linreg_cv))

## Handling missing data  --> No value for a feature in a particular row

# Shows feature
print(music_df.isna().sum().sort_values()) # Print the sum of columns where values are not a number

# 1. Drop missing data :)
music_df = music_df.dropna(subset=["genre", "popularity", "loudness", "liveness", "tempo"])
print(music_df.isna().sum().sort_values())

# 2. Imput values --> Make educated guess of values that are not present!
# Use mean, median, most frequent value. We must split our data to avoid data lakage :) --> LOOK DATA LAKAGE

X_cat = music_df["genre"].values.reshape(-1, 1)
X_num  = music_df.drop(["genre", "popularity"], axis=1).values
y = music_df["popularity"].values
X_train_cat, X_test_cat, y_train_cat, y_test_cat = train_test_split(X_cat, y, test_size=0.2, random_state=42)
X_train_num, X_test_num, y_train_num, y_test_num = train_test_split(X, y, test_size=0.2, random_state=42)

imp_cat = SimpleImputer(strategy="most_frequent")
X_train_cat = imp_cat.fit_transform(X_train_cat)
X_test_cat = imp_cat.transform(X_test_cat)

# Convert genre to a binary feature
music_df["genre"] = np.where(music_df["genre"] == "Rock", 1 0)

#For numeric data
imp_num = SimpleImputer() # By default, strategy = mean
X_train_num = imp_num.fit_transform(X_train_num)
X_test_num = imp_.num.transform(X_test_num)
X_train = np.append(X_train_num, X_train_cat, axis=1)
X_test = np.append(X_test_num, X_test_cat, axis=1)

# NOTE: IMPUTERS ARE KNOWN ALSO AS TRANSFORMERS :)

# Imputing within a pipeline
# To build a pipeline:
# Note: In a pipeline, every step must be a transformer except the last step, which is the model! :)
steps = [("imputation", SimpleImputer()),
          ("Logistic_regression", LogisticRegression())]
pipeline = Pipeline(steps)
X_train_cat, X_test_cat, y_train_cat, y_test_cat = train_test_split(X_cat, y, test_size=0.2, random_state=42)
pipeline.fit(X_train, y_train)
pipeline.score(X_test, y_test)
