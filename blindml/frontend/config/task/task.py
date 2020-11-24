import csv
import json
from pprint import pformat
from types import SimpleNamespace
from typing import Any, Union

import _jsonnet as jsonnet
from joblib import load, dump

from blindml.backend.buid_model_search_space import build_model_search_space
from blindml.backend.nni_helper import (
    run_nni,
    get_experiment_update,
    make_nni_experiment_config,
)
from blindml.backend.run import get_model
from blindml.backend.search.preprocessing.selection import select_features
from blindml.backend.search.preprocessing.transform import scale, get_splits
from blindml.backend.training.train import train, eval_model
from blindml.data.dataset import TabularDataset
from blindml.frontend.reporting.metrics import Metric, get_mse, get_r2
from blindml.util import dict_hash


class Task:
    _task_fp: str
    _task_capsule: SimpleNamespace
    _task_hash: str
    _task: SimpleNamespace
    _json_str: str
    _nni_experiment_config: dict
    _experiment_name: str
    _experiment_name_with_hash: str
    task_type: str
    user: str
    _data_path: str
    _data_set: Union[TabularDataset, Any]

    def __init__(self, task_fp) -> None:
        self._task_fp = task_fp
        self._json_str = jsonnet.evaluate_file(task_fp)
        self._task_hash = dict_hash(
            json.loads(self._json_str), omit_keys=["access_token"]
        )
        self._task_capsule = json.loads(
            self._json_str, object_hook=lambda d: SimpleNamespace(**d)
        )
        # task capsule has meta info (access key and dos and such)
        # TODO: probably need better naming
        self._task = self._task_capsule.task
        self.task_type = self._task.type
        self.user = self._task_capsule.user
        self._data_path = self._task.payload.data_path
        self._experiment_name = f"{self.user}s_experiment"
        self._experiment_name_with_hash = f"{self._experiment_name}_{self._task_hash}"
        if self._data_path.endswith(".csv"):
            all_columns = next(csv.reader(open(self._data_path, "r")))
            # XOR
            assert hasattr(self._task.payload, "X_cols") != hasattr(self._task.payload, "drop_cols")
            if hasattr(self._task.payload, "drop_cols"):
                X_cols = list(set(all_columns) - set(self._task.payload.drop_cols) - {self._task.payload.y_col})
            else:
                X_cols = self._task.X_cols

            self._data_set = TabularDataset(
                csv_fp=self._data_path,
                y_col=self._task.payload.y_col,
                X_cols=X_cols,
            )
            # TODO: this is a hack - we shouldn't be passing dataset to the search space
            # the data on demand module should find the relevant dataset
            search_space = build_model_search_space(
                task_type=self.task_type,
                data_path=self._data_path,
                y_col=self._task.payload.y_col,
                X_cols=X_cols,
            )
        else:
            raise Exception("unsupported dataset")

        # TODO: i had change this but now i can't find where this file is???
        # vim venv/nni/main.js
        # // const expId = createNew ? utils_1.uniqueString(8) : resumeExperimentId;
        # const expId = resumeExperimentId;
        self._nni_experiment_config = make_nni_experiment_config(
            self._experiment_name_with_hash, search_space
        )

    def search_for_model(self):
        # this will resume? if experiment already exists?
        run_nni(self._nni_experiment_config)

    def get_model_search_update(self):
        sorted_good_trials = get_experiment_update(self._nni_experiment_config)
        # resort by time
        metric_values = [s["finalMetricData"] for s in sorted_good_trials]
        if len(metric_values) == 0:
            print("no successfully trained models yet")
        return metric_values[::-1]

    def get_top_model(self):
        # TODO: massage this into something more useful
        sorted_good_trials = get_experiment_update(self._nni_experiment_config)
        top_trial = sorted_good_trials[0]
        return top_trial["finalMetricData"][0]["data"], top_trial["hyperParameters"]

    def train_best_model(self):
        sorted_good_trials = get_experiment_update(self._nni_experiment_config)
        hyper_parameters = sorted_good_trials[0]["hyperParameters"]
        model = get_model(hyper_parameters)

        X, y = self._data_set.get_data(dropna=True)
        # TODO: this should be part of model search
        X_scaled = scale(X)
        X_train, X_test, y_train, y_test = get_splits(X_scaled, y)
        X_selected_train, feat_idxs = select_features(X_train, y_train)

        model = train(X_selected_train, y_train, model)
        # TODO: stub
        y_pred = eval_model(X_test[:, feat_idxs], model)
        return {
            "model": model,
            "scores": {
                str(Metric.MSE): get_mse(y_test, y_pred),
                str(Metric.R2): get_r2(y_test, y_pred),
            },
            "hyper_parameters": hyper_parameters,
        }

    def save_model(self, model, f_dir):
        dump(model, f"{f_dir}/{self._experiment_name}.joblib")

    def load_model(self, f_dir):
        model = load(f"{f_dir}/{self._experiment_name}.joblib")
        return model

    def evaluate_best_model(self, model, X):
        return model.predict(X)

    # goal is to be readable
    def __str__(self):
        return pformat(json.loads(self._json_str), indent=2, width=40)


def parse_task_capsule(task_fp):
    return Task(task_fp)

