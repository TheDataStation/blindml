import os
import unittest

os.environ["NNI_PLATFORM"] = "unittest"
from nni.platform.test import init_params

from blindml.backend.run import main


class MyTestCase(unittest.TestCase):
    def test_run(self):
        params = {
            "parameter_id": 101,
            "parameter_source": "algorithm",
            "parameters": {
                "task_type": {
                    "_name": "regression",
                    "model": {
                        "_name": "SVR",
                        "kernel": "rbf",
                        "gamma": "scale",
                        "C": 1e-06,
                        "shrinking": False,
                    },
                },
                "data_path": "/Users/maksim/dev_projects/blindml/data/xtb-redox.csv",
                "y_col": "EA",
                "drop_cols": [
                    "inchi_key",
                    "wall_time_neutral",
                    "EA_wall_time",
                    "IP_wall_time",
                    "xyz_neutral",
                    "xyz_reduced",
                    "xyz_oxidized",
                    "smiles",
                    "inchi",
                ],
            },
            "parameter_index": 0,
        }

        init_params(params)
        main()


if __name__ == "__main__":
    unittest.main()
