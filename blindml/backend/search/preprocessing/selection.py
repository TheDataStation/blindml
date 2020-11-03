from nni.feature_engineering.gradient_selector import FeatureGradientSelector


def select_features(X_train, y_train):
    fgs = FeatureGradientSelector()
    fgs.fit(X_train, y_train)
    feat_idxs = fgs.get_selected_features()
    return X_train[:, feat_idxs], feat_idxs
