import os
import unittest

os.environ["NNI_PLATFORM"] = "unittest"
from nni.platform.test import init_params

from blindml.backend.run import main


class MyTestCase(unittest.TestCase):
    def test_run(self):
        params = {
            "parameter_id": 2,
            "parameter_source": "algorithm",
            "parameters": {
                "task_type": {
                    "_name": "regression",
                    "model": {
                        "_name": "NearestNeighborsRegressor",
                        "n_neighbors": 50.0,
                        "weights": "distance",
                        "p": 1,
                        "metric": "mahalanobis",
                    },
                },
                "data_path": "./data/Perovskite_Stability_with_features.csv",
                "y_col": "energy_above_hull (meV/atom)",
                "X_cols": [
                    "thermal conductivity_AB_avg",
                    "BCCefflatcnt_AB_avg",
                    "Bsite_First Ionization Potential (V)_max",
                    "Asite_GSmagmom_min",
                    "Bsite_electrical conductivity_weighted_avg",
                    "Bsite_At. Radius   (angstroms)_weighted_avg",
                    "electrical conductivity_AB_diff",
                    "Bsite_NpUnfilled_weighted_avg",
                    "host_Bsite0_at. wt.",
                    "Asite_At. Radius   (angstroms)_weighted_avg",
                    "covalent radius_AB_avg",
                    "Asite_Atomic Volume (cm\u00b3/mol)_max",
                    "Asite_BCCenergy_pa_min",
                    "host_Asite0_IsAlkali",
                    "Bsite_n_ws^third_min",
                    "Asite_BCCefflatcnt_range",
                    "First Ionization Potential (V)_AB_avg",
                    "Asite_shannon_radii_min",
                    "host_Bsite0_At. #",
                    "host_Asite0_Heat of Vaporization",
                    "host_Asite0_IsBCC",
                    "Bsite_At. Radius   (angstroms)_max",
                    "Bsite_Period_weighted_avg",
                    "BCCvolume_padiff_AB_avg",
                    "Number of elements",
                    "Asite_IsPnictide_weighted_avg",
                    "ICSDVolume_AB_avg",
                    "Asite_BCCvolume_pa_weighted_avg",
                    "Heat of Vaporization_AB_ratio",
                    "Asite_Atomic Radius (\u00c5)_max",
                    "Bsite_ BP (K)_max",
                    "Asite_n_ws^third_weighted_avg",
                    "at. wt._AB_diff",
                    "Asite_At. Radius   (angstroms)_max",
                    "Bsite_IsMetal_weighted_avg",
                    "shannon_radii_AB_avg",
                    "Density_AB_avg",
                    "Bsite_Second Ionization Potential  (V)_weighted_avg",
                    "Bsite_ BP (K)_weighted_avg",
                    "Bsite_MendeleevNumber_min",
                    "GSenergy_pa_AB_avg",
                    "Asite_BCCvolume_padiff_weighted_avg",
                    "Asite_BCCenergy_pa_max",
                    "Asite_At. Radius   (angstroms)_min",
                    "host_Bsite0_IsNoblegas",
                    "Bsite_NdUnfilled_weighted_avg",
                    "host_Asite0_NsValence",
                    "host_Asite0_OrbitalD",
                    "Asite_shannon_radii_range",
                    "Ionization Energy (kJ/mol)_AB_avg",
                    "Asite_NfValence_weighted_avg",
                    "Electron Affinity (kJ/mol)_AB_avg",
                    "host_Bsite0_IsHexagonal",
                    "Asite_IsRareEarth_weighted_avg",
                    "Asite_IsBoron_weighted_avg",
                    "BCCefflatcnt_AB_ratio",
                    "Asite_IsHalogen_weighted_avg",
                    "MendeleevNumber_AB_avg",
                    "Ionization Energy (kJ/mol)_AB_ratio",
                    "specific heat capacity_AB_diff",
                    "Bsite_Third Ionization Potential  (V)_max",
                    "host_Asite0_IsCubic",
                    "Asite_NfUnfilled_weighted_avg",
                    "Bsite_At. #_weighted_avg",
                    "Asite_IsAlkali_max",
                    "Atomic Volume (cm\u00b3/mol)_AB_avg",
                    "host_Bsite0_Ionization Energy (kJ/mol)",
                    "Bsite_IsMetal_max",
                    "Asite_Ionic Radius (angstroms)_max",
                    "num_of_atoms_host_Asite0",
                    "Asite_BCCenergydiff_min",
                ],
            },
            "parameter_index": 0,
        }

        init_params(params)
        main()


if __name__ == "__main__":
    unittest.main()
