import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
    RocCurveDisplay,
    PrecisionRecallDisplay
)
import seaborn as sns


def plot_confusion_matrix(y_true, y_pred, title="Confusion Matrix", save_path=None):
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Human", "LLM"],
                yticklabels=["Human", "LLM"])

    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")

    if save_path:
        plt.savefig(save_path)
    plt.show()


def plot_roc_curve(y_true, y_scores, title="ROC Curve", save_path=None):
    fpr, tpr, _ = roc_curve(y_true, y_scores)

    disp = RocCurveDisplay(fpr=fpr, tpr=tpr)
    disp.plot()

    plt.title(title)
    
    if save_path:
        plt.savefig(save_path)
    plt.show()


def plot_pr_curve(y_true, y_scores, title="Precision-Recall Curve", save_path=None):
    precision, recall, _ = precision_recall_curve(y_true, y_scores)

    disp = PrecisionRecallDisplay(precision=precision, recall=recall)
    disp.plot()

    plt.title(title)

    if save_path:
        plt.savefig(save_path)
    plt.show()
