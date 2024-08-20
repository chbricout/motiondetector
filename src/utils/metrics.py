"""Module used to define special metrics"""

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import seaborn as sb
from sklearn.metrics import balanced_accuracy_score


def separation_capacity(df: pd.DataFrame):

    train = df[df["mode"] == "train"]
    val = df[df["mode"] == "val"]

    x_train = train["pred"].to_numpy().reshape(-1, 1)
    y_train = train["label"].to_numpy()
    x_val = val["pred"].to_numpy().reshape(-1, 1)
    y_val = val["label"].to_numpy()
    n_clusters = int(max(x_val) + 1)
    print(n_clusters)

    model = DecisionTreeClassifier(
        max_leaf_nodes=n_clusters, max_depth=int(np.ceil(n_clusters / 2))
    )
    model.fit(x_train, y_train)
    accuracy = balanced_accuracy_score(y_val, model.predict(x_val))

    thresholds = list(sorted(model.tree_.threshold[: n_clusters - 1]))
    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(1, 1, 1)

    sb.swarmplot(x=x_val, hue=y_val, ax=ax)
    for thresh in thresholds:
        ax.axvline(x=thresh, color="r")
    ax.set_xlabel("Prediction metric")
    ax.legend(title="label")

    return accuracy, fig, thresholds
