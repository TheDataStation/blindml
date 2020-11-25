import matplotlib.pyplot as plt
import pandas as pd
from scipy.cluster import hierarchy
import numpy as np
from scipy.stats import spearmanr


def get_data_stats(data: pd.DataFrame):
    return data.describe(), data.corr()


def show_correlation(X, X_names):
    feat_names = np.array(X_names)
    corr = spearmanr(X).correlation
    nan_cols = np.all(np.isnan(corr), axis=0)  # axis is reduction dim
    nan_rows = np.all(np.isnan(corr), axis=1)  # axis is reduction dim
    feat_names = feat_names[~nan_rows]
    corr = corr[~nan_rows, :]
    corr = corr[:, ~nan_cols]
    corr_linkage = hierarchy.ward(corr)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 14))

    dendro = hierarchy.dendrogram(
        corr_linkage, labels=feat_names, ax=ax1, leaf_rotation=90
    )
    dendro_idx = np.arange(0, len(dendro["ivl"]))

    ax2.imshow(corr[dendro["leaves"], :][:, dendro["leaves"]])
    ax2.set_xticks(dendro_idx)
    ax2.set_yticks(dendro_idx)
    ax2.set_xticklabels(dendro["ivl"], rotation="vertical")
    ax2.set_yticklabels(dendro["ivl"])
    fig.tight_layout()
    plt.show()
