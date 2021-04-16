import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn
from sklearn.model_selection import train_test_split

from blindml.data.statistics import show_correlation


class TabularDataset:
    _df: pd.DataFrame
    _csv_fp: str
    _y_col: str
    # TODO: this is sloppy because it depends on ordering staying fixed
    # it should be some kind of spec with ordering included
    _X_cols: list
    _X: np.ndarray = None
    _y: np.ndarray = None
    _X_train: np.ndarray = None
    _y_train: np.ndarray = None
    _X_test: np.ndarray = None
    _y_test: np.ndarray = None
    _test_size: float = 0.20
    _dropna: bool = True

    def __init__(self, csv_fp, y_col, X_cols, dropna=True) -> None:
        self._csv_fp = csv_fp
        self._df = pd.read_csv(csv_fp)
        self._dropna = dropna
        self._y_col = y_col
        self._X_cols = X_cols

        # force these to categorical for dod integ prototype
        for column in self._df.columns:
            self._df[column] = self._df[column].astype('category')

        print(f"Dataframe is {self._df}")
        print(f"Dataframe types are {self._df.dtypes}")

    def get_data(self, dropna=False):
        if self._X is None or self._y is None:
            df = self._df[self._X_cols + [self._y_col]]
            if dropna:
                df = df.dropna(axis="index")
            self._X, self._y = split_df_X_y(self._y_col, df)
        return self._X, self._y

    def get_train_data(self):
        if self._X_train is None or self._y_train is None:
            X, y = self.get_data()
            self._X_train, self._X_test, self._y_train, self._y_test = get_splits(
                X, y, test_size=self._test_size
            )
        return self._X_train, self._y_train

    def get_test_data(self):
        if self._X_test is None or self._y_test is None:
            X, y = self.get_data()
            self._X_train, self._X_test, self._y_train, self._y_test = get_splits(
                X, y, test_size=self._test_size
            )
        return self._X_test, self._y_test

    def get_data_stats(self):
        return self._df.describe()

    def plot_feature_correlation(self):
        show_correlation(self._df[self._X_cols].values, self._X_cols)

    @property
    def df(self):
        return self._df

    @property
    def X_cols(self):
        return self._X_cols

    @property
    def y_col(self):
        return self._y_col


def split_df_X_y(y_col, df: pd.DataFrame):
    # X, y = df[list(set(df.columns.values) - {y_col})].values, df[y_col].values
    # potentially useful to avoid discarding df types that i've hacked in earlier?
    X, y = df[list(set(df.columns.values) - {y_col})], df[y_col]
    return X, y


def load_csv_data(csv_fp, extra_cols):
    df = pd.read_csv(csv_fp)
    df = df.drop(columns=extra_cols)
    df = df.dropna(axis="index")
    return df


def get_splits(X, y, test_size=0.20):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )
    return X_train, X_test, y_train, y_test
