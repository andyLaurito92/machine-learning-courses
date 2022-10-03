from sklearn.preprocessing import StandardScaler

# One important staff: Having data scale up :)

# With this, you can see the ranges of your data
print(music_df[["duration_ms", "loudness", "speechiness"]].describe())

# Why do we want to use feature scaling?
# Because ml model use distance to inform how wel we are preditcing
# If the features are on larger scales, these features can disproportionaly influence the model!!

# Example: KNN uses distances when making predictions :)

#How to scale our data?
# Substract the mean and divide by variance --> Standardization

# Can also substract the minimun and divide by the range

# Can also normalize so the data ranges from -1 to +1

X = music_df.drop("genre", axis=1).values
y = music_df["genre"].values
X_train_cat, X_test_cat, y_train_cat, y_test_cat = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandarScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print(np.mean(X), np.std(X))
print(np.mean(X_scaled), np.std(X_scaled))

steps = [('scaler', StandarScaler()),
         ('knn', KNeighborsClassifier(n_neighbors=6))]

pipeline = Pipeline(steps)
X_train_cat, X_test_cat, y_train_cat, y_test_cat = train_test_split(X, y, test_size=0.2, random_state=42)

knn_scaled = pipeline.fit(X_train, y_train)
y_pred = knn_scaled.predict(X_test)
print(knn_scaled.score(X_test, y_test))
