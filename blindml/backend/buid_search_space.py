import json
import os

HERE = os.path.split(__file__)[0]


def build_search_space(task_type):
    search_space = {"task_type": {"_type": "choice", "_value": []}}
    hps = json.load(open(f"{HERE}/search/model_search/hp.json"))
    models = json.load(open(f"{HERE}/search/model_search/model_select.json"))

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
    # TODO: hack
    fp = f"{HERE}/search_space.json"
    json.dump(search_space, open(fp, "w"), indent=2)


if __name__ == "__main__":
    print(__file__)
