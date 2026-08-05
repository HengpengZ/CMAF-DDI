"""Evaluation metrics used by CMAF-DDI."""

from __future__ import annotations

from typing import Dict

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: np.ndarray,
    task: str,
) -> Dict[str, float]:
    average = "macro" if task == "multiclass" else "binary"
    metrics = {
        "macro_precision": float(
            precision_score(y_true, y_pred, average=average, zero_division=0)
        ),
        "macro_recall": float(
            recall_score(y_true, y_pred, average=average, zero_division=0)
        ),
        "macro_f1": float(
            f1_score(y_true, y_pred, average=average, zero_division=0)
        ),
        "accuracy": float(accuracy_score(y_true, y_pred)),
    }
    if task == "binary":
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_score))
    return metrics
