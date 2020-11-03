import logging

from sklearn.gaussian_process.kernels import (
    WhiteKernel,
    RBF,
    Matern,
    RationalQuadratic,
    ExpSineSquared,
    DotProduct,
    PairwiseKernel,
)

log = logging.getLogger(__file__)

GP_KERNELS = {
    "white": WhiteKernel,
    "rbf": RBF,
    "matern": Matern,
    "rational_quad": RationalQuadratic,
    "exp_sine_squared": ExpSineSquared,
    "dot": DotProduct,
    "pair": PairwiseKernel,
}


def get_model_hps(params):
    if params["name"] in {"GaussianProcessRegressor", "GaussianProcessClassifier"}:
        kernel_cons = GP_KERNELS[params["kernel"]]
        # leaving open the possibility we might want parameterize
        # the kernels too
        params["kernel"] = kernel_cons()

    # we'll do more here soon
    return params
