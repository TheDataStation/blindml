from nni.platform.test import init_params

from blindml.backend.run import main

params = {
    "parameter_id": 171,
    "parameter_source": "algorithm",
    "parameters": {
        "task_type": {
            "_name": "regression",
            "model": {
                "_name": "NearestNeighborsRegressor",
                "n_neighbors": 80.0,
                "weights": "distance",
                "p": 1,
                "metric": "chebyshev",
            },
        }
    },
    "parameter_index": 0,
}


def test():
    init_params(params)
    main()
