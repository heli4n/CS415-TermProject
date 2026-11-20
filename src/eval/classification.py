mport numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    recall_score,
    confusion_matrix
)


def calculate_metrics(y_true, y_pred, labels=None):
    """
    Calculate classification metrics.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if len(y_true) != len(y_pred):
        raise ValueError("True and predicted labels must have equal lengths.")

    if len(y_true) == 0:
        raise ValueError("Input arrays cannot be empty.")

    if labels is None:
        labels = np.unique(np.concatenate([y_true, y_pred]))

    accuracy = accuracy_score(y_true, y_pred)

    macro_f1 = f1_score(
        y_true,
        y_pred,
        labels=labels,
        average="macro",
        zero_division=0
    )

    per_class_recall = recall_score(
        y_true,
        y_pred,
        labels=labels,
        average=None,
        zero_division=0
    )

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=labels
    )

    return {
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "per_class_recall": dict(
            zip(labels.tolist(), per_class_recall.tolist())
        ),
        "confusion_matrix": cm.tolist(),
        "labels": labels.tolist()
    }


def bootstrap_confidence_interval(
    y_true,
    y_pred,
    metric="accuracy",
    n_bootstrap=1000,
    confidence=0.95,
    seed=42
):
    """
    Calculate bootstrap confidence interval by resampling tracks.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if len(y_true) != len(y_pred):
        raise ValueError("True and predicted labels must have equal lengths.")

    if len(y_true) == 0:
        raise ValueError("Input arrays cannot be empty.")

    if metric not in ["accuracy", "macro_f1"]:
        raise ValueError("Metric must be 'accuracy' or 'macro_f1'.")

    if n_bootstrap < 1:
        raise ValueError("n_bootstrap must be positive.")

    if not 0 < confidence < 1:
        raise ValueError("Confidence must be between 0 and 1.")

    rng = np.random.default_rng(seed)
    n = len(y_true)
    scores = []

    labels = np.unique(np.concatenate([y_true, y_pred]))

    for i in range(n_bootstrap):
        indices = rng.choice(n, size=n, replace=True)

        true_sample = y_true[indices]
        pred_sample = y_pred[indices]

        if metric == "accuracy":
            score = accuracy_score(true_sample, pred_sample)
        else:
            score = f1_score(
                true_sample,
                pred_sample,
                labels=labels,
                average="macro",
                zero_division=0
            )

        scores.append(score)

    alpha = (1 - confidence) / 2

    lower = np.quantile(scores, alpha)
    upper = np.quantile(scores, 1 - alpha)

    return {
        "metric": metric,
        "confidence": confidence,
        "lower": float(lower),
        "upper": float(upper)
    }

