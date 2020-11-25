{
  user: 'max',
  access_token: '1234567',
  task: {
    type: 'regression',
    payload: {
      y_col: 'EA',
      drop_cols: [
        'inchi_key',
        'wall_time_neutral',
        'EA_wall_time',
        'IP_wall_time',
        'xyz_neutral',
        'xyz_reduced',
        'xyz_oxidized',
        'smiles',
        'inchi',
      ],
      data_path: "/Users/maksim/dev_projects/blindml/data/xtb-redox.csv"
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
