import pandas as pd

from blindml.backend.training.data import split_data


def load_csv_data(csv_fp, extra_cols):
    df = pd.read_csv(csv_fp)
    df.drop(columns=extra_cols, inplace=True)
    df.dropna(axis="index", inplace=True)
    return df


def load_logans_data():
    df = load_csv_data(
        "/Users/maksim/dev_projects/blindml/data/xtb-redox.csv",
        [
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
    )
    X, y = split_data("EA", df)
    return X, y
