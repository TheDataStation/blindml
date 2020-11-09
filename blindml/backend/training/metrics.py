from sklearn.metrics import mean_squared_error, r2_score


def get_mse(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    return float(mse)

def get_r2(y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    return r2
