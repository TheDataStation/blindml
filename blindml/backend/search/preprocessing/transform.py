from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def get_splits(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )
    return X_train, X_test, y_train, y_test


def scale(X):
    scaler = StandardScaler()
    return scaler.fit_transform(X)
