"""Module used to define special metrics"""

from matplotlib import pyplot as plt
from matplotlib.figure import Figure
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score
import seaborn as sb


def prediction_report(y_val, y_pred)->tuple[float, dict[int, float], plt.Figure]:
    cm_fig = plot_confusion_mat(y_val, y_pred)
    acc = balanced_accuracy_score(y_val, y_pred)
    per_class_accuracy = [f1_score(np.array(y_val) == i, np.array(y_pred) == i) for i in range(np.max(y_val)+1)]
    return acc, per_class_accuracy, cm_fig

def separation_capacity(
    df: pd.DataFrame, train_on: str = "train", test_on: str = "val"
) -> tuple[float, Figure, list[float], plt.Figure]:
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
    train = df[df["mode"] == train_on]
    val = df[df["mode"] == test_on]

    x_train = train["pred"].to_numpy()
    y_train = train["label"].to_numpy()
    x_val = val["pred"].to_numpy()
    y_val = val["label"].to_numpy()

    thresholds, _ = separation_capacity_train(
        X=x_train, y=y_train
    )

    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(1, 1, 1)

    sb.stripplot(x=val["pred"], hue=val["label"], ax=ax)
    for thresh in thresholds:
        ax.axvline(x=thresh, color="r")
    ax.set_xlabel("Prediction metric")
    ax.legend(title="label")

    val_pred = predict_thresholds(x_val, thresholds)
    acc, per_class_accuracy,cm_fig = prediction_report(y_val, val_pred)
    
    return acc, per_class_accuracy, thresholds,fig, cm_fig


def separation_capacity_train(X, y):
    best_thresholds = None
    best_accuracy = 0
    sorted_indices = np.argsort(X)
    X_sort = X[sorted_indices]
    y_sort = y[sorted_indices]

    midpoints = (X_sort[:-1] + X_sort[1:]) / 2
    # Try all pairs of unique values in X as potential thresholds
    for i,t1 in enumerate(midpoints[:-1]):
        for t2 in midpoints[i+1:]:
            
            y_pred = predict_thresholds(X_sort, (t1,t2))
            accuracy = balanced_accuracy_score(y_sort, y_pred)
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_thresholds = (t1, t2)

    return best_thresholds, best_accuracy

def predict_thresholds(X:np.ndarray, thresholds:tuple[float,float]):
    return np.where(X < thresholds[0], 0, np.where(X < thresholds[1], 1, 2))

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
