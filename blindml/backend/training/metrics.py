from sklearn.metrics import mean_squared_error


def get_mse(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    return float(mse)
