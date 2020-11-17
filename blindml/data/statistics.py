import matplotlib.pyplot as plt
import pandas as pd


def get_data_stats(data: pd.DataFrame):
    return data.describe(), data.corr()


def show_correlation(corr):
    plt.matshow(corr)
    plt.title("Correlation Matrix", fontsize=16)
    plt.colorbar()
    plt.show()
