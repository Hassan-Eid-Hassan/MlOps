"""
Data integrity tests, simplified from the original MLOpsPython `data_test.py`.
Same four checks (schema, bad schema, missing values, distribution drift),
just pointed at the local `data/` folder instead of an Azure ML path helper.
"""
import os

import numpy as np
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(__file__), os.pardir, "data")
EXPECTED_FEATURE_COLUMNS = 10  # everything except the target column "Y"

# Historical mean/std of the training set (features + target), used to
# flag distribution drift in incoming data.
HISTORICAL_MEAN = np.array(
    [
        -3.63962254e-16, 1.26972339e-16, -8.01646331e-16, 1.28856202e-16,
        -8.99230414e-17, 1.29609747e-16, -4.56397112e-16, 3.87573332e-16,
        -3.84559152e-16, -3.39848813e-16, 1.52133484e02,
    ]
)
HISTORICAL_STD = np.array(
    [
        4.75651494e-02, 4.75651494e-02, 4.75651494e-02, 4.75651494e-02,
        4.75651494e-02, 4.75651494e-02, 4.75651494e-02, 4.75651494e-02,
        4.75651494e-02, 4.75651494e-02, 7.70057459e01,
    ]
)
SHIFT_TOLERANCE = 3  # max tolerated relative change in mean/std


def _path(filename):
    return os.path.join(DATA_DIR, filename)


def test_schema_is_as_expected():
    df = pd.read_csv(_path("diabetes.csv"))
    assert df.shape[1] - 1 == EXPECTED_FEATURE_COLUMNS


def test_bad_schema_is_detected():
    df = pd.read_csv(_path("diabetes_bad_schema.csv"))
    assert df.shape[1] - 1 != EXPECTED_FEATURE_COLUMNS


def test_missing_values_are_detected():
    df = pd.read_csv(_path("diabetes_missing_values.csv"))
    assert df.isna().sum().sum() > 0


def test_distribution_drift_is_detected():
    df = pd.read_csv(_path("diabetes_bad_dist.csv"))
    mean = df.values.mean(axis=0)
    std = df.values.std(axis=0)
    mean_shift = np.abs(mean - HISTORICAL_MEAN) > SHIFT_TOLERANCE * np.abs(HISTORICAL_MEAN)
    std_shift = np.abs(std - HISTORICAL_STD) > SHIFT_TOLERANCE * np.abs(HISTORICAL_STD)
    assert mean_shift.any() or std_shift.any()
