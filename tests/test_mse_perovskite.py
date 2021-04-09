import time
from blindml.frontend.config.task.task import parse_task_capsule
from sklearn.metrics import mean_squared_error

def test_perovskite_quick():
    """Tests that the perovskite demo has mean squared error less than a
    hardcoded threshold. If deliberate changes mean that this threshold
    needs to change, then it should be changed here, but only if the change
    is intentional.
    """
    task = parse_task_capsule("tests/perovskite_task.jsonnet")
    task.search_for_model()

    model = task._auto_sk_model

    X_train, y_train = task._data_set.get_train_data()

    y_pred = model.predict(X_train)

    mse = mean_squared_error(y_train, y_pred)

    """
    On @benclifford laptop, running for these times gives MSEs around these numbers:

     5 mins => 820
    15 mins => 591
    60 mins => 492

    but for now this test is kept deliberately short and inaccurate
    """

    # for 60 seconds testing. Parameterise for long and short tests (eg 1h vs 60s)
    threshold_mse = 1523

    if mse > threshold_mse:
        raise RuntimeError(f"Mean standard error {mse} was above reasonable threshold {threshold_mse}")

    print(f"test finished with mse {mse}")


if __name__ == "__main__":
    test_perovskite_quick()
    print("test successful")
