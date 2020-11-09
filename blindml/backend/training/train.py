import inspect
from enum import Enum, auto

from blindml.backend.training.metrics import get_mse


class Metric(Enum):
    def _generate_next_value_(name, start, count, last_values):
        return name

    MSE = auto()
    MAE = auto()
    ACCURACY = auto()


def train(X, y, model):
    if "sklearn" in inspect.getfile(model.__class__):
        model.fit(X, y)
    else:
        raise Exception("unsupported model")
    return model


def eval_model(X, y_true, model, metric=Metric.MSE):
    if "sklearn" in inspect.getfile(model.__class__):
        y_pred = model.predict(X)
    else:
        raise Exception("unsupported model")
    if metric == Metric.MSE:
        return y_true, get_mse(y_true, y_pred)
    else:
        raise Exception("unsupported metric")
