{
  user: 'max',
  access_token: '1234567',
  task: {
    type: 'regression',
    payload: {
      y_col: 'med_v',
      drop_cols: [
        'black'
      ],
      data_path: "/home/maksim/dev_projects/blindml/data/housing_small.csv"
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
