from nni.feature_engineering.gradient_selector import FeatureGradientSelector
import numpy as np

def select_features(X_train, y_train):
    try:
        fgs = FeatureGradientSelector(n_epochs=10, device="cuda")
        fgs.fit(X_train, y_train)
        feat_idxs = fgs.get_selected_features()
    except AssertionError as a:
        assert a.args[0] == 'Torch not compiled with CUDA enabled'
        fgs = FeatureGradientSelector(n_epochs=10, device="cpu")
        fgs.fit(X_train, y_train)
        feat_idxs = fgs.get_selected_features()

    if not feat_idxs:
        _rows, cols = X_train.shape
        feat_idxs = np.arange(cols)
    return X_train[:, feat_idxs], feat_idxs
