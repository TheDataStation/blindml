import pandas as pd


def load_logans_data():
    df = pd.read_csv("/Users/maksim/dev_projects/blindml/data/xtb-redox.csv")
    df.drop(
        columns=[
            "inchi_key",
            "wall_time_neutral",
            "EA_wall_time",
            "IP_wall_time",
            "xyz_neutral",
            "xyz_reduced",
            "xyz_oxidized",
        ],
        inplace=True,
    )
    df.drop(columns=["smiles", "inchi"], inplace=True)
    # df.set_index("inchi", inplace=True)
    df.dropna(axis="index", inplace=True)
    y_col = "EA"
    X, y = df[list(set(df.columns.values) - {y_col})].values, df[y_col].values
    return X, y
