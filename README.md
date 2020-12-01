- [Structure](#structure)
- [Entry point](#entry-point)
- [Requirements](#requirements)
- [TODO](#todo)

# Structure

<img width="1237" alt="image" src="https://user-images.githubusercontent.com/5657668/97810686-63ec2280-1c3b-11eb-8624-fef46da8e568.png">

# Entry point

[demo notebook](demo.ipynb)

# Requirements

Don't update `requirements.txt`; instead update `requirements.in` and run `pip-compile` using

```
pip-compile --extra-index-url https://pypi.anaconda.org/scipy-wheels-nightly/simple
```

Note that you will need to use the same flag when installing:

```
pip install -r requirements.txt --extra-index-url https://pypi.anaconda.org/scipy-wheels-nightly/simple
```

Also if you run into an error about `tensorflow==2.3.1` then you need to `pip install --upgrade pip`.

## What-if

```
jupyter nbextension install --py --symlink --sys-prefix witwidget
jupyter nbextension enable --py --sys-prefix witwidget
```

# TODO

spinners for long running "explanation" functions.
