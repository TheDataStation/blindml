import json
import os

# reading this from the same directory as the .py file is
# probably not the right thing to do when installing as a package.
# HERE = os.path.split(__file__)[0]
# so for now, look in the current directory we are running from,
# which means you can only run from the blindml source directory.
# which is ick.
HERE = "."


def build_model_search_space(task_type: str, data_path: str, **kwargs):
    search_space = {
        "task_type": {"_type": "choice", "_value": []},
        "data_path": {"_type": "choice", "_value": [data_path]},
    }
    if data_path.endswith(".csv"):
        assert "y_col" in kwargs, "need to specify y_col for csv"
        search_space["y_col"] = {"_type": "choice", "_value": [kwargs["y_col"]]}
        if "X_cols" in kwargs:
            search_space["X_cols"] = {"_type": "choice", "_value": [kwargs["X_cols"]]}
        elif "drop_cols" in kwargs:
            search_space["drop_cols"] = {
                "_type": "choice",
                "_value": [kwargs["drop_cols"]],
            }
        else:
            raise Exception("need to specify either X_cols or drop_cols")

    hps = json.load(open(f"{HERE}/hp.json"))
    models = json.load(open(f"{HERE}/model_select.json"))

    classifier_choices_with_hps = []
    for classifier in models["classification"]:
        classifier_choices_with_hps.append({"_name": classifier, **hps[classifier]})
    regressor_choices_with_hps = []
    for regressor in models["regression"]:
        regressor_choices_with_hps.append({"_name": regressor, **hps[regressor]})

    if task_type == "classification":
        search_space["task_type"]["_value"].append(
            {
                "_name": "classification",
                "model": {"_type": "choice", "_value": classifier_choices_with_hps},
            }
        )
    elif task_type == "regression":
        search_space["task_type"]["_value"].append(
            {
                "_name": "regression",
                "model": {"_type": "choice", "_value": regressor_choices_with_hps},
            }
        )
    else:
        raise Exception(f"unsupported task type {task_type}")
    return search_space


if __name__ == "__main__":
    print(__file__)
