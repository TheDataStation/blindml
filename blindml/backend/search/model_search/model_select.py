import logging

from sklearn.gaussian_process import GaussianProcessRegressor, GaussianProcessClassifier
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import (
    LinearRegression,
    Ridge,
    RidgeClassifier,
    Lasso,
    Lars,
    ElasticNet,
    LogisticRegression,
    ARDRegression,
)
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

log = logging.getLogger(__file__)


def get_model_cons(params):
    model_dict = {
        "classification": {
            "DecisionTreeClassifier": DecisionTreeClassifier,
            "GaussianProcessClassifier": GaussianProcessClassifier,
            "NearestNeighborsClassifier": KNeighborsClassifier,
            "RidgeClassifier": RidgeClassifier,
            "SVC": SVC,
        },
        "regression": {
            "ARDRegression": ARDRegression,
            "DecisionTreeRegressor": DecisionTreeRegressor,
            "ElasticNet": ElasticNet,
            "GaussianProcessRegressor": GaussianProcessRegressor,
            "KernelRidgeRegression": KernelRidge,
            "Lars": Lars,
            "Lasso": Lasso,
            "LinearRegression": LinearRegression,
            "LogisticRegression": LogisticRegression,
            "NearestNeighborsRegressor": KNeighborsRegressor,
            "RidgeRegression": Ridge,
            "SVR": SVR,
        },
    }
    task_type = params["task_type"]["name"]
    if not model_dict.get(task_type):
        log.exception(f"Not supported task {task_type}")
        exit(1)

    model_type = params["task_type"]["model"]["name"]
    if not model_dict[task_type].get(model_type):
        log.exception(f"Not supported model {model_type}")
        exit(1)

    model = model_dict[task_type][model_type]

    return model
