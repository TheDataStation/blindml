import time
from blindml.frontend.config.task.task import parse_task_capsule
from sklearn.metrics import mean_squared_error

def test_perovskite():
    """Tests that the perovskite demo has mean squared error less than a
    hardcoded threshold. If deliberate changes mean that this threshold
    needs to change, then it should be changed here, but only if the change
    is intentional.
    """
    task = parse_task_capsule("tests/perovskite_task.jsonnet")
    task.search_for_model()
    while not task.get_model_search_update():
        print("no model trained yet")
        time.sleep(5)

    model = task.train_best_model()

    X_train, y_train = task._data_set.get_train_data()

    y_pred = model.predict(X_train)

    mse = mean_squared_error(y_train, y_pred)

    raise RuntimeError(f"BENC: MSE threshold not implemented - {mse}")
