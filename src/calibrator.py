"""Model calibration via Platt scaling and calibration metrics."""

import numpy as np

from feature_engineer import FEATURE_NAMES, extract_feature_matrix, features_to_array
from label_generator import generate_labels


def calibrate_model(model, trials, method="sigmoid"):
    """Apply Platt scaling calibration.

    Args:
        model: Trained classifier.
        trials: List of trial dicts.
        method: Calibration method ("sigmoid" for Platt, "isotonic").

    Returns:
        Calibrated classifier, or original model if calibration fails.
    """
    from sklearn.calibration import CalibratedClassifierCV

    feature_dicts = extract_feature_matrix(trials)
    X = np.array(features_to_array(feature_dicts, FEATURE_NAMES), dtype=np.float64)
    labels = generate_labels(trials)
    y = np.array([lb["completed"] for lb in labels], dtype=np.int32)

    try:
        cal = CalibratedClassifierCV(model, method=method, cv=2)
        cal.fit(X, y)
        return cal
    except Exception:
        return model


def calibration_metrics(model, trials):
    """Compute Brier score and calibration statistics.

    Args:
        model: Trained classifier with predict_proba.
        trials: List of trial dicts.

    Returns:
        Dict with brier_score, n_samples, mean_predicted, mean_actual.
    """
    from sklearn.metrics import brier_score_loss

    feature_dicts = extract_feature_matrix(trials)
    X = np.array(features_to_array(feature_dicts, FEATURE_NAMES), dtype=np.float64)
    labels = generate_labels(trials)
    y = np.array([lb["completed"] for lb in labels], dtype=np.int32)

    probs = model.predict_proba(X)
    prob_col = 1 if probs.shape[1] > 1 else 0
    p = probs[:, prob_col]

    brier = float(brier_score_loss(y, p))

    return {
        "brier_score": brier,
        "n_samples": len(y),
        "mean_predicted": float(np.mean(p)),
        "mean_actual": float(np.mean(y)),
    }
