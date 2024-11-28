"""Module used to define special metrics"""

import numpy as np
import seaborn as sb
from matplotlib import pyplot as plt
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score


def prediction_report(y_val, y_pred) -> tuple[float, dict[int, float], plt.Figure]:
    cm_fig = plot_confusion_mat(y_val, y_pred)
    acc = balanced_accuracy_score(y_val, y_pred)
    per_class_accuracy = [
        f1_score(np.array(y_val) == i, np.array(y_pred) == i)
        for i in range(np.max(y_val) + 1)
    ]
    return acc, per_class_accuracy, cm_fig


def plot_confusion_mat(y_val, val_pred):
    cm = confusion_matrix(y_val, val_pred)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(1, 1, 1)
    sb.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Pred 0", "Pred 1", "Pred 2"],
        yticklabels=["True 0", "True 1", "True 2"],
        ax=ax,
    )

    # Add labels and title
    ax.set_ylabel("Actual")
    ax.set_xlabel("Predicted")
    ax.set_title("Confusion Matrix")
    return fig
