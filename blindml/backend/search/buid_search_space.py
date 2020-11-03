import json

SEARCH_SPACE = {"task_type": {"_type": "choice", "_value": []}}

if __name__ == "__main__":
    hps = json.load(open("model_search/hp.json"))
    models = json.load(open("model_search/model_select.json"))

    classifier_choices_with_hps = []
    for classifier in models["classification"]:
        classifier_choices_with_hps.append({"_name": classifier, **hps[classifier]})
    regressor_choices_with_hps = []
    for regressor in models["regression"]:
        regressor_choices_with_hps.append({"_name": regressor, **hps[regressor]})

    # SEARCH_SPACE["task_type"]["_value"].append(
    #     {
    #         "_name": "classification",
    #         "model": {"_type": "choice", "_value": classifier_choices_with_hps},
    #     }
    # )
    SEARCH_SPACE["task_type"]["_value"].append(
        {
            "_name": "regression",
            "model": {"_type": "choice", "_value": regressor_choices_with_hps},
        }
    )
    json.dump(SEARCH_SPACE, open("search_space.json", "w"), indent=2)
