- [Structure](#structure)
- [Entry point](#entry-point)
- [Requirements](#requirements)
- [TODO](#todo)

# Structure

```
blindml/
├── backend
│   ├── buid_model_search_space.py
│   ├── nni_helper.py
│   ├── run.py
│   ├── search
│   │   ├── model_search
│   │   │   ├── hp.json
│   │   │   ├── hp_search.py
│   │   │   ├── model_select.json
│   │   │   └── model_select.py
│   │   └── preprocessing
│   │       ├── selection.py
│   │       └── transform.py
│   ├── search_space.json
│   └── training
│       └── train.py
├── data
│   ├── dataset.py
│   └── statistics.py
├── frontend
│   ├── config
│   │   └── task
│   │       └── task.py
│   ├── reporting
│   │   └── metrics.py
│   └── results
│       ├── api.py
│       └── labels.py
├── runner.py
└── util.py
```

<img width="1237" alt="image" src="https://user-images.githubusercontent.com/5657668/97810686-63ec2280-1c3b-11eb-8624-fef46da8e568.png">

# Entry point

Execute `./run.sh` to find start `nni` and find parameters for an `sklearn` model.

# Requirements

Don't update `requirements.txt`; instead update `requirements.in` and run `pip-compile`.

# TODO

Right now only backend is (sort of) implemented. Frontend is still in the offing.