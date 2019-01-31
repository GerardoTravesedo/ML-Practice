from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd


class BudgetCleaner(BaseEstimator, TransformerMixin):

    def fit(self, *_):
        return self

    def transform(self, dataframe, *_):
        # Budget column has object type. Almost all the values are numbers,
        # but some might be strings with letters.
        # This converts the values to a number, emptying those that are
        # not numbers
        budget_with_numbers = \
          pd.to_numeric(dataframe['budget'], errors='coerce')
        temp_budget = dataframe.drop(columns=['budget'])
        temp_budget['budget'] = budget_with_numbers

        # We also want to keep only those that have a valid budget. That means
        # a budget over 0 in our case
        clean_budget = temp_budget[temp_budget['budget'] > 100000]

        return clean_budget
