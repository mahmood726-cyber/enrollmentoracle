"""Predict completion probability for a single trial."""

import numpy as np

from feature_engineer import FEATURE_NAMES, extract_features, features_to_array


def predict_single(model, trial, feature_names=None):
    """Predict completion probability for one trial.

    Args:
        model: Trained classifier with predict_proba.
        trial: Trial dict.
        feature_names: Ordered feature names. Defaults to FEATURE_NAMES.

    Returns:
        Dict with nctId, probability, risk (LOW/MEDIUM/HIGH), features.
    """
    if feature_names is None:
        feature_names = FEATURE_NAMES

    feats = extract_features(trial)
    X = np.array(features_to_array([feats], feature_names), dtype=np.float64)
    probs = model.predict_proba(X)
    prob_col = 1 if probs.shape[1] > 1 else 0
    p_complete = float(probs[0, prob_col])

    if p_complete >= 0.7:
        risk = "LOW"
    elif p_complete >= 0.4:
        risk = "MEDIUM"
    else:
        risk = "HIGH"

    return {
        "nctId": trial.get("nctId", ""),
        "probability": p_complete,
        "risk": risk,
        "features": feats,
    }
