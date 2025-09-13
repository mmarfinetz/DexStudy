"""
Evaluation utilities for the DEX valuation study.
Provides functions to compute error metrics on the original USD scale and to produce residual diagnostics.
"""
import numpy as np
import pandas as pd
from typing import Dict


def inverse_log_transform(log_vals: np.ndarray) -> np.ndarray:
    """Inverse of the natural log transformation."""
    return np.exp(log_vals)


def compute_metrics(y_true_log: np.ndarray, y_pred_log: np.ndarray) -> Dict[str, float]:
    """Compute evaluation metrics on the USD scale.

    Args:
        y_true_log: True target values (log scale).
        y_pred_log: Predicted target values (log scale).

    Returns:
        Dictionary with RMSE (USD millions), MAE (USD millions), MAPE, R2, and directional accuracy.
    """
    y_true = inverse_log_transform(y_true_log)
    y_pred = inverse_log_transform(y_pred_log)
    residuals = y_true - y_pred
    rmse = np.sqrt(np.mean(residuals ** 2))
    mae = np.mean(np.abs(residuals))
    mape = np.mean(np.abs(residuals) / np.maximum(y_true, 1e-9))
    r2 = 1 - np.sum(residuals ** 2) / np.sum((y_true - np.mean(y_true)) ** 2)
    directional_accuracy = np.mean(np.sign(np.diff(y_true)) == np.sign(np.diff(y_pred)))
    return {
        'rmse_usd': float(rmse),
        'mae_usd': float(mae),
        'mape': float(mape),
        'r2': float(r2),
        'directional_accuracy': float(directional_accuracy),
    }
