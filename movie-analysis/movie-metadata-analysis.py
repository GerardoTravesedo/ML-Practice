import pandas as pd
import numpy as np
from GenreEncoder import GenreEncoder
from BudgetCleaner import BudgetCleaner
from RuntimeCleaner import RuntimeCleaner
from PopularityCleaner import PopularityCleaner
from RevenueCleaner import RevenueCleaner
from sklearn.pipeline import Pipeline
from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor


possible_genres = ["Action", "Adventure", "Animation", "Comedy", "Crime",
                   "Documentary", "Drama", "Family", "Fantasy", "Foreign",
                   "History", "Horror", "Music", "Mystery", "Romance",
                   "Science Fiction", "TV Movie", "Thriller", "War", "Western"]


def read_dataset():
    movies = pd.read_csv("dataset/movies_metadata.csv")
    return movies[
      ['budget', 'popularity', 'genres', 'revenue']
    ]

dataset_interesting_fields = read_dataset()

pipeline = Pipeline(
  steps=[("Clean budget", BudgetCleaner()),
         ("Clean popularity", PopularityCleaner()),
         # ("Clean runtime", RuntimeCleaner()),
         ("Clean revenue", RevenueCleaner()),
         ("Encode genres", GenreEncoder(possible_genres))])

clean_dataset = pipeline.transform(dataset_interesting_fields)
clean_dataset.to_csv("dataset/test.csv")

training_labels = clean_dataset['revenue']
training_instances = clean_dataset.drop(columns=['revenue'])

# reg = linear_model.LinearRegression()
reg = RandomForestRegressor()
# Decision trees really overfit the data
# reg = DecisionTreeRegressor()
reg.fit(training_instances, training_labels)

corr_matrix = clean_dataset.corr()
print(corr_matrix['revenue'].sort_values(ascending=False))

# 187436818.0
print(reg.predict(
  np.array([60000000.0, 17.924927, 1.0,
            0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 1.0, 0.0, 0.0]).reshape(1, -1)))

# 122195920
print(reg.predict(
  np.array([18000000.0, 10.870138, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0]).reshape(1, -1)))
