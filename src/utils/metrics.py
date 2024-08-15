"""Module used to define special metrics"""

from collections.abc import Sequence

from matplotlib import pyplot as plt
from matplotlib.figure import Figure
import numpy as np
from sklearn.cluster import KMeans
import seaborn as sb
from sklearn.metrics import balanced_accuracy_score


def separation_capacity(
    label: Sequence[int], pred: Sequence[float]
) -> tuple[float, Figure, list[float]]:
    """Aim to measure how well we are able to separate our continous
      predictions to predict our classes with n_classes -1 thresholds.
      We simply train a KMeans algorithm on the whole dataset and get
      accuracy on the whole dataset.

    Args:
        label (Sequence[int]): Sequence of ground truth labels
        pred (Sequence[float]): Sequence of predictions on continuous
        metrics (SSIM or MOTION modes)

    Returns:
        tuple[float, Figure, list[float]]: accuracy using thresholds,
        swarm plot using thresholds, list of thresholds
    """
    n_clusters = int(max(label) + 1)
    pred_np = np.array(pred).reshape(-1, 1)
    model = KMeans(n_clusters=n_clusters)
    model.fit(pred_np, label)
    accuracy = balanced_accuracy_score(label, model.predict(pred_np))
    centers = model.cluster_centers_.flatten()
    centers = np.sort(centers)  # Sort centers to make threshold extraction easier
    # The threshold is midway through both centroid
    thresholds = [(centers[i] + centers[i + 1]) / 2 for i in range(len(centers) - 1)]

    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(1, 1, 1)

    sb.swarmplot(x=pred, hue=label, ax=ax)
    for thresh in thresholds:
        ax.axvline(x=thresh, color="r")
    ax.set_xlabel("Prediction metric")
    ax.legend(title="label")

    return accuracy, fig, thresholds
