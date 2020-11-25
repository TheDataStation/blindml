from enum import Enum, auto

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score


def get_mse(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    return float(mse)


def get_r2(y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    return r2


def plot_trial_record(metric_values):
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    for m in set(metric_values[0].keys()) - {"default"}:
        splot = sns.scatterplot(
            x=range(1, len(metric_values) + 1),
            y=[mm[m] for mm in metric_values],
            label=m,
            legend=True,
            ax=ax
        )
    splot.set(title="Model search results")
    splot.set(xlabel="trials")
    splot.set(ylabel="metric")
    splot.set(yscale="log")
    # splot.set(legend=True)
    for m in set(metric_values[0].keys()) - {"default"}:
        min_x, min_y = len(metric_values) + 1, metric_values[-1][m]
        plt.scatter(min_x, min_y, marker="o", s=100)
        plt.annotate(
            f"{min_y:.3e}", (min_x, min_y), xytext=(-40, 10), textcoords="offset points"
        )

    plt.show()


class Metric(Enum):
    def _generate_next_value_(name, start, count, last_values):
        return name

    MSE = auto()
    MAE = auto()
    ACCURACY = auto()
    R2 = auto()
