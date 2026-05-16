"""Train XGBoost/GradientBoosting classifier with stratified CV."""

import warnings
import numpy as np

from feature_engineer import FEATURE_NAMES, extract_feature_matrix, features_to_array
from label_generator import generate_labels


def _get_classifier(seed=42):
    """Get the best available gradient boosting classifier.

    Tries XGBoost first, falls back to sklearn GradientBoostingClassifier.

    Returns:
        (classifier_instance, classifier_name)
    """
    try:
        from xgboost import XGBClassifier
        clf = XGBClassifier(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.1,
            random_state=seed,
            use_label_encoder=False,
            eval_metric="logloss",
        )
        return clf, "XGBoost"
    except ImportError:
        from sklearn.ensemble import GradientBoostingClassifier
        clf = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.1,
            random_state=seed,
        )
        return clf, "GradientBoosting"


def train_model(trials, n_folds=3, seed=42):
    """Train a completion prediction model with stratified K-fold CV.

    Args:
        trials: List of trial dicts.
        n_folds: Number of CV folds. Adjusted down if too few samples per class.
        seed: Random seed for reproducibility.

    Returns:
        (model, metrics_dict) where metrics has:
            - accuracy: mean CV accuracy
            - auc: mean AUC-ROC (None if not computable)
            - feature_names: list of feature names used
            - classifier_type: name of the classifier
            - n_samples: total samples
            - n_completed: count of completed trials
            - n_failed: count of terminated/withdrawn trials
    """
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.metrics import roc_auc_score

    # Extract features and labels
    feature_dicts = extract_feature_matrix(trials)
    X = np.array(features_to_array(feature_dicts, FEATURE_NAMES), dtype=np.float64)
    labels = generate_labels(trials)
    y = np.array([lb["completed"] for lb in labels], dtype=np.int32)

    n_completed = int(np.sum(y == 1))
    n_failed = int(np.sum(y == 0))

    # Adjust n_folds so each class has at least 2 samples per fold
    min_class_count = min(n_completed, n_failed)
    effective_folds = min(n_folds, min_class_count)
    effective_folds = max(effective_folds, 2)  # minimum 2 folds

    clf, clf_name = _get_classifier(seed)

    # Stratified CV
    skf = StratifiedKFold(n_splits=effective_folds, shuffle=True, random_state=seed)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cv_scores = cross_val_score(clf, X, y, cv=skf, scoring="accuracy")

    # Try AUC
    auc_val = None
    if n_completed > 0 and n_failed > 0:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                auc_scores = cross_val_score(clf, X, y, cv=skf, scoring="roc_auc")
                auc_val = float(np.mean(auc_scores))
        except Exception:
            auc_val = None

    # Train final model on all data
    clf_final, _ = _get_classifier(seed)
    clf_final.fit(X, y)

    metrics = {
        "accuracy": float(np.mean(cv_scores)),
        "auc": auc_val,
        "feature_names": FEATURE_NAMES[:],
        "classifier_type": clf_name,
        "n_samples": len(y),
        "n_completed": n_completed,
        "n_failed": n_failed,
        "cv_folds": effective_folds,
    }

    return clf_final, metrics


def evaluate_model(model, trials):
    """Evaluate a trained model on trials.

    Args:
        model: Trained classifier with predict_proba method.
        trials: List of trial dicts.

    Returns:
        List of dicts with nctId, prob_complete, predicted, actual.
    """
    feature_dicts = extract_feature_matrix(trials)
    X = np.array(features_to_array(feature_dicts, FEATURE_NAMES), dtype=np.float64)
    labels = generate_labels(trials)

    probs = model.predict_proba(X)
    # prob_complete is probability of class 1
    prob_col = 1 if probs.shape[1] > 1 else 0

    results = []
    for i, trial in enumerate(trials):
        results.append({
            "nctId": trial.get("nctId", ""),
            "prob_complete": float(probs[i, prob_col]),
            "predicted": int(probs[i, prob_col] >= 0.5),
            "actual": labels[i]["completed"],
        })

    return results
