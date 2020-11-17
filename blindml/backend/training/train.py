import inspect


def train(X, y, model):
    if "sklearn" in inspect.getfile(model.__class__):
        model.fit(X, y)
    else:
        raise Exception("unsupported model")
    return model


def eval_model(X, model):
    if "sklearn" in inspect.getfile(model.__class__):
        y_pred = model.predict(X)
        return y_pred
    else:
        raise Exception("unsupported model")
