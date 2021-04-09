import time
from blindml.frontend.config.task.task import parse_task_capsule
from sklearn.metrics import accuracy_score

def test_big_small_classifier_quick():
    """Tests that the perovskite demo has mean squared error less than a
    hardcoded threshold. If deliberate changes mean that this threshold
    needs to change, then it should be changed here, but only if the change
    is intentional.
    """
    task = parse_task_capsule("tests/big_small_classifier.jsonnet")
    task.search_for_model()

    model = task._auto_sk_model

    X_train, y_train = task._data_set.get_train_data()

    y_pred = model.predict(X_train)

    accuracy = accuracy_score(y_train, y_pred)

    # on such a tiny dataset, this should predict with complete
    # accuracy on the training data
    threshold_accuracy = 1.0

    if accuracy < threshold_accuracy:
        raise RuntimeError(f"Classifier accuracy {accuracy} was below reasonable threshold {threshold_accuracy}")

    print(f"test finished with accuracy {accuracy}")

def test_big_small_classifier_explicit_X_cols():
    """Same test as above, but with a config that uses explicit X cols,
    to test that configuration code path"""
    task = parse_task_capsule("tests/big_small_classifier_explicit_xcols.jsonnet")
    task.search_for_model()

    model = task._auto_sk_model

    X_train, y_train = task._data_set.get_train_data()

    y_pred = model.predict(X_train)

    accuracy = accuracy_score(y_train, y_pred)

    # on such a tiny dataset, this should predict with complete
    # accuracy on the training data
    threshold_accuracy = 1.0

    if accuracy < threshold_accuracy:
        raise RuntimeError(f"Classifier accuracy {accuracy} was below reasonable threshold {threshold_accuracy}")

    print(f"test finished with accuracy {accuracy}")


if __name__ == "__main__":
    test_big_small_classifier_quick()
    print("test successful")
