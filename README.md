- [Structure](#structure)
- [Requirements](#requirements)

# Structure

```
blindml
├── backend
│   ├── search
│   │   ├── data_search.py
│   │   ├── model_search
│   │   │   ├── arch_search.py
│   │   │   ├── hp_search.py
│   │   │   └── model_select.py
│   │   └── preprocessing
│   │       ├── selection.py
│   │       └── transform.py
│   └── training
│       ├── distributed.py
│       ├── hpo.py
│       ├── metrics.py
│       └── optim.py
├── frontend
│   ├── config
│   │   ├── data
│   │   │   ├── labels.py
│   │   │   └── samples.py
│   │   └── task
│   │       ├── budget.py
│   │       ├── expectations.py
│   │       └── objective.py
│   ├── reporting
│   │   ├── curves.py
│   │   └── statistics.py
│   └── results
│       ├── api.py
│       └── labels.py
└── runner.py

```

<img width="1237" alt="image" src="https://user-images.githubusercontent.com/5657668/97810686-63ec2280-1c3b-11eb-8624-fef46da8e568.png">


# Requirements

Don't update `requirements.txt`; instead update `requirements` and run `pip-compile requirements`.