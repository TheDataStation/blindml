import json
import logging
from numbers import Number

import nni

from blindml.backend.search.data_search import load_logans_data
from blindml.backend.search.model_search.hp_search import get_model_hps
from blindml.backend.search.model_search.model_select import get_model_cons
from blindml.backend.search.preprocessing.selection import select_features
from blindml.backend.search.preprocessing.transform import scale, get_splits
from blindml.backend.training.train import train, eval_model

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


def main():
    params = nni.get_next_parameter()
    model = get_model(params)

    X, y = load_logans_data()
    X_scaled = scale(X)
    X_train, X_test, y_train, y_test = get_splits(X_scaled, y)
    X_selected_train, feat_idxs = select_features(X_train, y_train)

    model = train(X_selected_train, y_train, model)
    score = eval_model(X_test[:, feat_idxs], y_test, model)
    nni.report_final_result(score)


if __name__ == "__main__":
    main()
