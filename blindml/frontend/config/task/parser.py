import json

import _jsonnet as jsonnet
from glom import glom

from blindml.backend.search.data_search import load_csv_data


def parse_task_capsule(task_fp):
    json_str = jsonnet.evaluate_file(task_fp)
    task = json.loads(json_str)

    csv_fp = glom(task, 'task.payload.data')
    y_col = glom(task, 'task.payload.y_col')
    extra_cols = glom(task, 'task.payload.extra_cols')
    df = load_csv_data(csv_fp, extra_cols)
    return df, y_col
