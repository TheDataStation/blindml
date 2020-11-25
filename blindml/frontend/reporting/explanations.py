from sklearn.inspection import permutation_importance
from matplotlib import pyplot as plt
import numpy as np


def get_perm_feat_import(clf, X_val, y_val, X_cols):
    feature_names = np.array(X_cols)
    result = permutation_importance(clf, X_val, y_val, n_repeats=30)
    perm_sorted_idx = result.importances_mean.argsort()

    fig, ax2 = plt.subplots(1, 1, figsize=(12, 8))
    ax2.boxplot(
        result.importances[perm_sorted_idx].T,
        vert=False,
        labels=feature_names[perm_sorted_idx],
    )
    fig.tight_layout()
    plt.show()
