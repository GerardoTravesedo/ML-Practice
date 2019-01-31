from sklearn.base import BaseEstimator, TransformerMixin


class RuntimeCleaner(BaseEstimator, TransformerMixin):

    def fit(self, *_):
        return self

    def transform(self, dataframe, *_):
        return dataframe[dataframe['runtime'] > 0]
