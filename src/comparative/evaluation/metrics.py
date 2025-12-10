# src/comparative/evaluation/metrics.py

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, mean_squared_error, mean_absolute_error
)
import numpy as np
from typing import Literal, Dict, Any, Sequence, Union

# Allowed average types for sklearn metrics
AverageType = Literal['micro', 'macro', 'samples', 'weighted', 'binary']

def compute_classification_metrics(
    y_true: Sequence, 
    y_pred: Sequence, 
    average: AverageType = "weighted"
) -> Dict[str, Any]:
    """Returns common classification metrics as a dict."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average=average, zero_division=0),
        "recall": recall_score(y_true, y_pred, average=average, zero_division=0),
        "f1": f1_score(y_true, y_pred, average=average, zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred)
    }

def compute_regression_metrics(
    y_true: Sequence[float], 
    y_pred: Sequence[float]
) -> Dict[str, float]:
    """Returns regression metrics."""
    return {
        "mse": mean_squared_error(y_true, y_pred),
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred)))
    }
