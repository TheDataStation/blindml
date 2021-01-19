from nni.feature_engineering.gradient_selector import FeatureGradientSelector
import numpy as np
import torch

def select_features(X_train, y_train):
    # Prefer to run with CUDA but fall back if it detects that that will
    # not work - there are two ways this detection is done: if the code
    # raises an AssertionError coming from torch not supporting CUDA,
    # and from Torch supporting CUDA but CUDA not being available on the
    # executing system.
    if torch.cuda.is_available():
        try:
            fgs = FeatureGradientSelector(n_epochs=10, device="cuda")
        except AssertionError as a:
            assert a.args[0] == 'Torch not compiled with CUDA enabled'
            fgs = FeatureGradientSelector(n_epochs=10, device="cpu")
    else:
        fgs = FeatureGradientSelector(n_epochs=10, device="cpu")

    fgs.fit(X_train, y_train)
    feat_idxs = fgs.get_selected_features()

    if not feat_idxs:
        _rows, cols = X_train.shape
        feat_idxs = np.arange(cols)
    return feat_idxs
