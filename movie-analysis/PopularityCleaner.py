from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd


class PopularityCleaner(BaseEstimator, TransformerMixin):

    def fit(self, *_):
        return self

    def transform(self, dataframe, *_):
        popularity_with_numbers = \
          pd.to_numeric(dataframe['popularity'], errors='coerce')
        clean_popularity = dataframe.drop(columns=['popularity'])
        clean_popularity['popularity'] = popularity_with_numbers

        return clean_popularity
