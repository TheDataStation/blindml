{
  user: 'max',
  access_token: '1234567',
  task: {
    type: 'classification',
    search_time: 60,
    payload: {
      y_col: 'human_size',
      drop_cols: [],
      data_path: "./tests/big_small_classifier.csv"
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
