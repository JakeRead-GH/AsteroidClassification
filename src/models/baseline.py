import numpy as np
import pandas as pd

class BaselineModel:
    """Predicts the majority class from the training set."""
    def __init__(self):
        self.majority_class = None

    def fit(self, y_train):
        self.majority_class = y_train.value_counts().idxmax()
        print(f"Majority class determined: {self.majority_class}")

    def predict(self, X_test):
        if self.majority_class is None:
            raise ValueError("Fit the model before predicting.")
        return pd.Series([self.majority_class] * len(X_test), index=X_test.index)
