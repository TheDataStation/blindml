import csv
import numpy as np
import json
from pprint import pformat
from types import SimpleNamespace
from typing import Any, Union

import _jsonnet as jsonnet
from IPython.core.display import display
from joblib import load, dump
from witwidget import WitConfigBuilder, WitWidget

from blindml.backend.buid_model_search_space import build_model_search_space
from blindml.backend.nni_helper import (
    run_nni,
    get_experiment_update,
    make_nni_experiment_config,
)
from blindml.backend.run import get_model
from blindml.backend.search.preprocessing.selection import select_features
from blindml.backend.training.train import train
from blindml.data.dataset import TabularDataset
from blindml.frontend.reporting.explanations import (
    get_perm_feat_import,
    plot_partial_dep,
    plot_feat_import,
    get_very_important_features,
)
from blindml.frontend.reporting.metrics import plot_trial_record
from blindml.frontend.reporting.wit import df_to_examples, custom_predict
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
            all_columns = next(
                csv.reader(open(self._data_path, "r", encoding="utf-8-sig"))
            )
            # XOR
            assert hasattr(self._task.payload, "X_cols") != hasattr(
                self._task.payload, "drop_cols"
            )
            if hasattr(self._task.payload, "drop_cols"):
                X_cols = list(
                    set(all_columns)
                    - set(self._task.payload.drop_cols)
                    - {self._task.payload.y_col}
                )
            else:
                X_cols = self._task.X_cols

            self._data_set = TabularDataset(
                csv_fp=self._data_path, y_col=self._task.payload.y_col, X_cols=X_cols
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

        # TODO: this is a dirty hack in order to have a fixed experiment name in both blindml and nni
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
        # re-sort by time
        metric_values = [s["finalMetricData"] for s in sorted_good_trials]
        if len(metric_values) == 0:
            print("no successfully trained models yet")
        return metric_values[::-1]

    def get_best_model(self):
        sorted_good_trials = get_experiment_update(self._nni_experiment_config)
        hyper_parameters = sorted_good_trials[0]["hyperParameters"]
        model = get_model(hyper_parameters)
        return model

    def train_best_model(self):
        model = self.get_best_model()

        # # TODO: this should be part of model search
        # X_scaled = scale(X)

        X_train, y_train = self._data_set.get_train_data()
        feat_idxs = select_features(X_train, y_train)
        X_selected_train = X_train[:, feat_idxs]

        model = train(X_selected_train, y_train, model)
        # TODO: include or don't the metrics in the serialization?
        # y_pred = eval_model(X_test[:, feat_idxs], model)
        return model

    def get_explanations(self, model=None):
        if model is None:
            model = self.train_best_model()

        print("trial record")
        self.plot_trial_record()
        try:
            print("feature correlations")
            self.plot_feature_correlations()
        except:
            pass
        try:
            print("feature importances")
            self.plot_feature_importance(model)
        except:
            pass

        try:
            print("partial dependences and individual conditional expectation")
            self.plot_partial_dependence(model)
        except:
            pass

    def get_wit(self, model=None):
        if model is None:
            model = self.train_best_model()

        df = self._data_set._df
        X_cols, y_col = self._data_set._X_cols, self._data_set._y_col
        features_and_labels = X_cols + [y_col]
        # examples = df_to_examples(df)
        # feature_spec = create_feature_spec(df, features_and_labels)
        # create_feature_columns(X_cols, feature_spec)

        num_datapoints = 1000
        test_examples = df_to_examples(
            self._data_set._df[features_and_labels][0:num_datapoints]
        )
        config_builder = (
            WitConfigBuilder(
                test_examples[:num_datapoints], feature_names=features_and_labels
            )
            # TODO: the task should be aware of itself (i.e. what kind of task it is)
            .set_model_type("regression")
            .set_custom_predict_fn(
                lambda examples: custom_predict(model, X_cols, examples)
            )
            .set_target_feature(y_col)
        )

        display(WitWidget(config_builder))

    def plot_feature_correlations(self):
        self._data_set.plot_feature_correlation()

    def plot_feature_importance(self, model):
        X_test, y_test = self._data_set.get_test_data()
        # TODO: this should take into account the fact that feature selection ran
        # this is a len(feat)*5 ndarray where each element is basically the change
        # in score due to permuting that feature with another
        # large differences in scores implies high importance
        # WARNING: if you permute with a colinear feature then you'll get zero importance
        # in theory we should PCA first
        importances = get_perm_feat_import(model, X_test, y_test)
        plot_feat_import(importances, self._data_set._X_cols)

    def plot_partial_dependence(self, model):
        X_test, y_test = self._data_set.get_test_data()
        plot_partial_dep(model, X_test, y_test, self._data_set._X_cols)

    def plot_trial_record(self):
        metric_values = self.get_model_search_update()
        plot_trial_record(metric_values)

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
