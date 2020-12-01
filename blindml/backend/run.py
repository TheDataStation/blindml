import csv
import json
import logging
from numbers import Number

import nni

from blindml.backend.search.model_search.hp_search import get_model_hps
from blindml.backend.search.model_search.model_select import get_model_cons
from blindml.backend.search.preprocessing.selection import select_features
from blindml.backend.search.preprocessing.transform import scale
from blindml.backend.training.train import train, eval_model
from blindml.data.dataset import TabularDataset, get_splits
from blindml.frontend.reporting.metrics import get_mse, get_r2

log = logging.getLogger(__file__)


def replace_name(params):
    return json.loads(json.dumps(params).replace("_name", "name"))


def intify(params):
    for k, v in params.items():
        if isinstance(v, Number) and int(v) - v == 0:
            params[k] = int(v)
    return params


def get_model(params):
    params = replace_name(params)
    model_cons = get_model_cons(params)
    params = get_model_hps(params["task_type"]["model"])
    params.pop("name")
    # this isn't so straightforward
    params = intify(params)
    model = model_cons(**params)
    return model


def get_dataset(params) -> TabularDataset:
    data_path = params["data_path"]
    if data_path.endswith(".csv"):
        all_columns = next(csv.reader(open(data_path, "r", encoding='utf-8-sig')))
        # XOR
        assert ("X_cols" in params) != ("drop_cols" in params)
        if "drop_cols" in params:
            X_cols = list(
                set(all_columns)
                - set(params["drop_cols"])
                - {params["y_col"]}
            )
        else:
            X_cols = params["X_cols"]

        data_set = TabularDataset(
            csv_fp=data_path, y_col=params["y_col"], X_cols=X_cols
        )

    else:
        raise Exception(f"unsupported data set {data_path}")
    return data_set


def main():
    params = nni.get_next_parameter()
    model = get_model(params)
    data_set = get_dataset(params)
    X, y = data_set.get_data(dropna=True)

    # TODO: this shouldn't be done - this should be part of search
    X_scaled = scale(X)
    X_train, X_test, y_train, y_test = get_splits(X_scaled, y)
    feat_idxs = select_features(X_train, y_train)
    X_selected_train = X_train[:, feat_idxs]

    model = train(X_selected_train, y_train, model)
    y_pred = eval_model(X_test[:, feat_idxs], model)
    mse_score = get_mse(y_test, y_pred)
    r2_score = get_r2(y_test, y_pred)
    nni.report_final_result({
        "mse": mse_score,
        "default": mse_score,
        "r2": r2_score,
    })


if __name__ == "__main__":
    main()
