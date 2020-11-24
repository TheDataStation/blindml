from sklearn.inspection import permutation_importance


def get_perm_feat_import(model, X_val, y_val):
    r = permutation_importance(model, X_val, y_val, n_repeats=30)

    for i in r.importances_mean.argsort()[::-1]:
        if r.importances_mean[i] - 2 * r.importances_std[i] > 0:
            print(
                # f"{diabetes.feature_names[i]:<8}"
                f"{r.importances_mean[i]:.3f}"
                f" +/- {r.importances_std[i]:.3f}"
            )
