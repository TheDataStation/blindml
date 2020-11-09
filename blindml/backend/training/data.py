import pandas as pd


def split_data(y_col, df: pd.DataFrame):
    X, y = df[list(set(df.columns.values) - {y_col})].values, df[y_col].values
    return X, y
