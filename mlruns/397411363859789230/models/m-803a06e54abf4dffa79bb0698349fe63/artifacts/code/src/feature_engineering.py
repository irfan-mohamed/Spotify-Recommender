from sklearn.base import BaseEstimator, TransformerMixin


class FeatureEngineer(BaseEstimator, TransformerMixin):

    def __init__(self):
        pass

    def fit(self, X):
        return self

    def transform(self, X):
        X = X.copy()

        X["energy_dance_ratio"] = X["energy"] / (X["danceability"] + 1e-5)
        X["acoustic_softness"] = X["acousticness"] * (1 - X["energy"])
        X["vocal_presence"] = 1 - X["instrumentalness"]

        return X
