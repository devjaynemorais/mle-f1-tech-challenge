# Encoding Strategy City - Frequency Encoding
from sklearn.base import BaseEstimator, TransformerMixin

class FrequencyEncoder(BaseEstimator, TransformerMixin):

    def __init__(self, column):
        self.column = column
        self.freq_map = None

    def fit(self, X, y=None):
        self.freq_map = X[self.column].value_counts(normalize=True)
        return self

    def transform(self, X):
        X = X.copy()
        X[f"{self.column}_freq"] = X[self.column].map(self.freq_map)
        return X