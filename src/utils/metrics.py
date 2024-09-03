"""Module used to define special metrics"""

from matplotlib import pyplot as plt
from matplotlib.figure import Figure
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import balanced_accuracy_score
import seaborn as sb


def separation_capacity(df: pd.DataFrame) -> tuple[float, Figure, list[float]]:
    """Determine the separation capacity by
    training the simplest Tree Classifier on
    train data and then prediction on validation

    Args:
        df (pd.DataFrame): Dataframe zith columns:
            pred, label and mode

    Returns:
        tuple[float,Figure,list[float]]: accuracy,
        Figure to save and list of thresholds to classify
    """
    train = df[df["mode"] == "train"]
    val = df[df["mode"] == "val"]

    x_train = train["pred"].to_numpy()
    y_train = train["label"].to_numpy()
    x_val = val["pred"].to_numpy()
    y_val = val["label"].to_numpy()
    accuracy, model = separation_capacity_tree(
        X_train=x_train, y_train=y_train, X_val=x_val, y_val=y_val
    )

    thresholds = list(
        sorted(filter(lambda x: x > -2.0, model.tree_.threshold), reverse=True)
    )

    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(1, 1, 1)

    sb.stripplot(x=val["pred"], hue=val["label"], ax=ax)
    for thresh in thresholds:
        ax.axvline(x=thresh, color="r")
    ax.set_xlabel("Prediction metric")
    ax.legend(title="label")

    return accuracy, fig, thresholds


def separation_capacity_tree(
    X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray
) -> tuple[float, DecisionTreeClassifier]:

    if len(X_train.shape) == 1:
        X_train = X_train.reshape(-1, 1)
    if len(X_val.shape) == 1:
        X_val = X_val.reshape(-1, 1)

    n_clusters = max(int(max(y_train) + 1), 2)

    model = DecisionTreeClassifier(
        max_leaf_nodes=n_clusters,
        max_depth=int(np.ceil(n_clusters / 2)),
        class_weight="balanced",
    )
    model.fit(X_train, y_train)
    accuracy = balanced_accuracy_score(y_val, model.predict(X_val))
    return accuracy, model
