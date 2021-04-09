{
  user: 'max',
  access_token: '1234567',
  task: {
    type: 'regression',
    search_time: 60,
    payload: {
      y_col: 'energy_above_hull (meV/atom)',
      drop_cols: [
        'formation_energy (eV/atom)',
        'Material Composition',
        'A site #1',
        'A site #2',
        'A site #3',
        'B site #1',
        'B site #2',
        'B site #3',
        'X site',
      ],
      data_path: "./data/Perovskite_Stability_with_features.csv"
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
