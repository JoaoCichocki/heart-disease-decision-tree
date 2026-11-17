"""Download and prepare the Cleveland Heart Disease dataset."""

import pandas as pd

DATA_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "heart-disease/processed.cleveland.data"
)

FEATURE_NAMES = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal"
]
TARGET_NAME = "target"


def load_heart_disease_data(url=DATA_URL):
    """
    Load the Cleveland dataset, remove incomplete rows and convert the
    original target (0..4) into a binary target (0 or 1).
    """
    columns = FEATURE_NAMES + [TARGET_NAME]
    data = pd.read_csv(url, names=columns, na_values="?")
    data = data.dropna().astype(float)
    data[TARGET_NAME] = (data[TARGET_NAME] > 0).astype(int)

    X = data[FEATURE_NAMES].to_numpy()
    y = data[TARGET_NAME].to_numpy(dtype=int)
    return X, y, FEATURE_NAMES
