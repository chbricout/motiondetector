"""Module used to define special metrics"""

from collections.abc import Sequence

from matplotlib import pyplot as plt
from matplotlib.figure import Figure
import numpy as np
from sklearn.tree import DecisionTreeClassifier
import seaborn as sb
from sklearn.metrics import balanced_accuracy_score



def separation_capacity(label, pred) : 
    n_clusters = int(max(label) + 1)
    print(n_clusters)

    x = np.asarray(pred).reshape(-1,1)

    model = DecisionTreeClassifier(max_leaf_nodes=n_clusters, max_depth=int(np.ceil(n_clusters/2)))
    model.fit(x, label)
    accuracy = balanced_accuracy_score(label,model.predict(x))
    
    thresholds = list(sorted(model.tree_.threshold[:n_clusters-1]))
    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(1, 1, 1)

    sb.swarmplot(x=pred, hue=label, ax=ax)
    for thresh in thresholds:
        ax.axvline(x=thresh, color="r")
    ax.set_xlabel("Prediction metric")
    ax.legend(title="label")

    return accuracy, fig, thresholds
