from sklearn.preprocessing import StandardScaler


def scale(X):
    scaler = StandardScaler()
    return scaler.fit_transform(X)
