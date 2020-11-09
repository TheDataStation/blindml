import json
from pprint import pformat
from types import SimpleNamespace

import _jsonnet as jsonnet
from joblib import dump, load

from blindml.backend.buid_search_space import build_search_space
from blindml.backend.nni_helper import (
    run_nni,
    get_experiment_update,
    make_nni_experiment_config,
)
from blindml.backend.run import get_model
from blindml.backend.search.data_search import load_logans_data
from blindml.backend.search.preprocessing.selection import select_features
from blindml.backend.search.preprocessing.transform import scale, get_splits
from blindml.backend.training.metrics import get_mse, get_r2
from blindml.backend.training.train import train, eval_model, Metric


class Task:
    _task_fp: str
    _task: SimpleNamespace
    _json_str: str
    _nni_experiment_config: dict
    _experiment_name: str
    task_type: str
    user: str

    def __init__(self, task_fp) -> None:
        self._task_fp = task_fp
        self._json_str = jsonnet.evaluate_file(task_fp)
        # TODO: this is a hack; should design real task class
        self._task = json.loads(
            self._json_str, object_hook=lambda d: SimpleNamespace(**d)
        )
        self.task_type = self._task.task.type
        self.user = self._task.user
        self._experiment_name = f"{self.user}s_experiment"
        self._nni_experiment_config = make_nni_experiment_config(self._experiment_name)

    def run(self):
        # self.__experiment_name = f"{self.user}_{''.join(random.sample(string.ascii_letters + string.digits, 8))}"
        build_search_space(task_type=self.task_type)
        self._nni_experiment_config = run_nni(experiment_name=self._experiment_name)

    def get_experiment_update(self):
        top_trial = get_experiment_update(self._nni_experiment_config)
        return top_trial

    def train_best_model(self):
        top_trial = get_experiment_update(self._nni_experiment_config)
        model = get_model(top_trial["hyperParameters"])

        X, y = load_logans_data()
        X_scaled = scale(X)
        X_train, X_test, y_train, y_test = get_splits(X_scaled, y)
        X_selected_train, feat_idxs = select_features(X_train, y_train)

        model = train(X_selected_train, y_train, model)
        # TODO: stub
        y_pred = eval_model(X_test[:, feat_idxs], model)
        return (
            model,
            {
                str(Metric.MSE): get_mse(y_test, y_pred),
                str(Metric.R2): get_r2(y_test, y_pred),
            },
        )

    def save_best_model(self, f_dir):
        model, scores = self.train_best_model()
        dump(model, f"{f_dir}/{self._experiment_name}.joblib")
        with open(f"{f_dir}/{self._experiment_name}_top_score.json", "w") as f:
            json.dump(scores, f)

    def load_best_model(self, f_dir):
        self.save_best_model(f_dir)
        model = load(f"{f_dir}/{self._experiment_name}.joblib")
        with open(f"{f_dir}/{self._experiment_name}_top_score.json", "r") as f:
            scores = json.load(f)
        return model, scores

    def evaluate_best_model(self, f_dir, X):
        model, score = self.load_best_model(f_dir)
        return model.predict(X)

    # goal is to be readable
    def __str__(self):
        return pformat(json.loads(self._json_str), indent=2, width=40)


def parse_task_capsule(task_fp):
    return Task(task_fp)

    # csv_fp = glom(task, 'task.payload.data')
    # y_col = glom(task, 'task.payload.y_col')
    # extra_cols = glom(task, 'task.payload.extra_cols')
    # df = load_csv_data(csv_fp, extra_cols)
    # return df, y_col
