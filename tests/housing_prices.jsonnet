{
  user: 'max',
  access_token: '1234567',
  task: {
    type: 'regression',
    payload: {
      y_col: 'SalePrice',
      X_cols: [
        'LotArea',
        'OverallQual',
        'OverallCond',
        'YearBuilt',
        'YearRemodAdd',
        'BsmtFinSF1',
        'BsmtUnfSF',
        'TotalBsmtSF',
        '1stFlrSF',
        '2ndFlrSF',
        'GrLivArea',
        'GarageArea',
        'GarageCars',
        'WoodDeckSF',
        'OpenPorchSF',
        'EnclosedPorch',
      ],
      data_path: "/home/maksim/dev_projects/blindml/data/housing_prices_small.csv"
    },
  },
  dos: {
    metric: 'accuracy',
    range: [0.8, 1],
  },
  trust_constraints: {
    freshness: 'last_week',
    user: 'all_groups',
  },
}
