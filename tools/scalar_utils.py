"""Shared scaler utilities for training and prediction."""


class NoScaler:
    """Return input data unchanged through a scikit-learn-compatible interface."""

    def fit(self, X, y=None):
        return self

    def fit_transform(self, X):
        return X

    def transform(self, X):
        return X
