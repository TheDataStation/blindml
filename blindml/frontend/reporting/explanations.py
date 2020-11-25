from sklearn.inspection import (
    permutation_importance,
    plot_partial_dependence,
    partial_dependence,
)
from matplotlib import pyplot as plt
import numpy as np


def get_perm_feat_import(model, X_vals, y_vals) -> np.ndarray:
    # The permutation importance of a feature is calculated as follows.
    # First, a baseline metric, defined by scoring, is evaluated on a
    # (potentially different) dataset defined by the X.
    # Next, a feature column from the validation set is permuted
    # and the metric is evaluated again. The permutation importance
    # is defined to be the difference between the baseline metric and
    # metric from permutating the feature column.
    result = permutation_importance(model, X_vals, y_vals, n_repeats=5)
    return result.importances


def get_very_important_features(importances: np.ndarray, threshold=0.05):
    importance_means = importances.mean(axis=1)
    perm_sorted_idx = importance_means.argsort()
    very_important_idxs = perm_sorted_idx[importance_means > threshold]
    return very_important_idxs


def plot_feat_import(importances: np.ndarray, feature_names: list):
    feature_names = np.array(feature_names)
    very_important_idxs = get_very_important_features(importances)
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.boxplot(
        importances[very_important_idxs].T,
        vert=False,
        labels=feature_names[very_important_idxs],
    )
    fig.tight_layout()
    plt.show()


def plot_partial_dep(model, X_vals: np.ndarray, y_vals: np.ndarray, feat_names: list):
    importances = get_perm_feat_import(model, X_vals, y_vals)
    very_important_idxs = get_very_important_features(importances)
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    display = plot_partial_dependence(
        model,
        X_vals,
        very_important_idxs[-5:],
        feature_names=feat_names,
        kind="both",
        subsample=50,
        n_jobs=3,
        grid_resolution=20,
        random_state=0,
        ax=ax
    )
    display.figure_.subplots_adjust(hspace=0.5)
    plt.show()
