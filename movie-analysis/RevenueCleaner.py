from sklearn.base import BaseEstimator, TransformerMixin


class RevenueCleaner(BaseEstimator, TransformerMixin):

    def fit(self, *_):
        return self

    def transform(self, dataframe, *_):
        return dataframe[dataframe['revenue'] > 100000]
