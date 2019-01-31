from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import json
import numpy as np


class GenreEncoder(BaseEstimator, TransformerMixin):

    def __init__(self, possible_genres):
        self.possible_genres = possible_genres

    def fit(self, *_):
        return self

    def transform(self, dataframe, *_):
        # Get 'genres' column/series
        genres = dataframe['genres']
        dropped_genres = dataframe.drop(columns=['genres'])

        # Convert the series from string to json
        genres_json = \
            genres.apply(lambda x: json.loads(str(x).replace("'", "\"")))

        # Get pandas dataframe with 1s only in the columns that represent the
        # genres of the movie
        #
        # We are traversing the genre series/column and generating a new series
        # per element with one element per genre, which means that we have a
        # series of series, which is a dataframe
        genres_multiple_hot_encoding = \
            genres_json.apply(lambda x: self._get_multiple_hot_encoding(x))

        # Merging two dataframes
        # Axis = 1 means that we are adding columns
        # Axis = 0 (default) means that we are adding rows
        return pd.concat([dropped_genres, genres_multiple_hot_encoding], axis=1)

    # 1) Generate an array of 0s and 1s for the movie. 0 if the movie doesn't
    #    have the index genre and 1 if it does
    # 2) Generate a series element with the array build, using all the possible
    #    movie genres as the index
    # Return: The function returns a series
    def _get_multiple_hot_encoding(self, movie_genres):
        number_genres = len(self.possible_genres)

        movie_genre_encoding = np.zeros(number_genres)

        for genre in movie_genres:
            genre_index = self.possible_genres.index(genre['name'])
            movie_genre_encoding[genre_index] = 1

        return pd.Series(movie_genre_encoding, index=self.possible_genres)
