import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn


class TabularDataset:
    _df: pd.DataFrame
    _csv_fp: str
    _y_col: str
    _X_cols: list
    __drop_cols: list
    _X: np.ndarray = None
    _y: np.ndarray = None

    def __init__(self, csv_fp, y_col, X_cols) -> None:
        self._csv_fp = csv_fp
        self._df = pd.read_csv(csv_fp)

        self._y_col = y_col
        self._X_cols = X_cols

    def get_data(self, dropna=False):
        if (self._X is None) or (self._y is None):
            df = self._df[self._X_cols + [self._y_col]]
            if dropna:
                df = df.dropna(axis="index")
            self._X, self._y = split_df_X_y(self._y_col, df)
        return self._X, self._y

    def get_data_stats(self):
        return self._df.describe()

    def show_feature_correlation(self):
        corr = self._df[self._X_cols].corr()
        sn.heatmap(corr, annot=True)
        plt.show()


def split_df_X_y(y_col, df: pd.DataFrame):
    X, y = df[list(set(df.columns.values) - {y_col})].values, df[y_col].values
    return X, y


def load_csv_data(csv_fp, extra_cols):
    df = pd.read_csv(csv_fp)
    df = df.drop(columns=extra_cols)
    df = df.dropna(axis="index")
    return df
