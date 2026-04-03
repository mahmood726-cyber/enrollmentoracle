"""Advanced statistical engine for EnrollmentOracle.

11 methods across two layers:
  Layer 1 (publication-essential): bootstrap CI, calibration slope,
    learning curves, DeLong test, SMOTE, hyperparameter tuning.
  Layer 2 (methodologically novel): competing risks, conformal prediction,
    tree SHAP, Bayesian model averaging, survival prediction.
"""

import math
import warnings

import numpy as np
from scipy import stats as sp_stats
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    learning_curve as sklearn_learning_curve,
)

from feature_engineer import FEATURE_NAMES, extract_feature_matrix, features_to_array
from label_generator import generate_labels


# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------


def _extract_Xy(trials):
    """Extract feature matrix X and binary label vector y from trials."""
    feature_dicts = extract_feature_matrix(trials)
    X = np.array(features_to_array(feature_dicts, FEATURE_NAMES), dtype=np.float64)
    labels = generate_labels(trials)
    y = np.array([lb["completed"] for lb in labels], dtype=np.int32)
    return X, y, labels


def _get_nct_ids(trials):
    """Return list of nctId strings."""
    return [t.get("nctId", "") for t in trials]


# ===================================================================
#  Layer 1 --- Publication-Essential (6 methods)
# ===================================================================


# 1. Bootstrap Confidence Intervals -----------------------------------

def bootstrap_ci(y_true, y_pred_proba, metric_fn, n_bootstrap=1000,
                 alpha=0.05, seed=42):
    """Bootstrap confidence interval for any metric.

    Args:
        y_true: 1-D array of true binary labels.
        y_pred_proba: 1-D array of predicted probabilities.
        metric_fn: callable(y_true, y_pred_proba) -> float.
        n_bootstrap: number of resamples.
        alpha: significance level (default 0.05 -> 95 % CI).
        seed: random seed.

    Returns:
        dict with keys: estimate, ci_lower, ci_upper, se.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred_proba = np.asarray(y_pred_proba, dtype=np.float64)
    n = len(y_true)

    rng = np.random.RandomState(seed)
    estimate = float(metric_fn(y_true, y_pred_proba))

    boot_vals = []
    for _ in range(n_bootstrap):
        idx = rng.choice(n, n, replace=True)
        # Need both classes in the resample for AUC-type metrics
        try:
            val = metric_fn(y_true[idx], y_pred_proba[idx])
            boot_vals.append(val)
        except Exception:
            continue

    if len(boot_vals) == 0:
        return {"estimate": estimate, "ci_lower": estimate,
                "ci_upper": estimate, "se": 0.0}

    boot_vals = np.array(boot_vals)
    ci_lower = float(np.percentile(boot_vals, alpha / 2 * 100))
    ci_upper = float(np.percentile(boot_vals, (1 - alpha / 2) * 100))
    se = float(np.std(boot_vals, ddof=1))

    return {"estimate": estimate, "ci_lower": ci_lower,
            "ci_upper": ci_upper, "se": se}


def bootstrap_auc_ci(y_true, y_pred_proba, n_bootstrap=1000, alpha=0.05,
                     seed=42):
    """Bootstrap CI for AUC-ROC."""
    return bootstrap_ci(y_true, y_pred_proba, roc_auc_score,
                        n_bootstrap=n_bootstrap, alpha=alpha, seed=seed)


def bootstrap_brier_ci(y_true, y_pred_proba, n_bootstrap=1000, alpha=0.05,
                       seed=42):
    """Bootstrap CI for Brier score."""
    return bootstrap_ci(y_true, y_pred_proba, brier_score_loss,
                        n_bootstrap=n_bootstrap, alpha=alpha, seed=seed)


def bootstrap_accuracy_ci(y_true, y_pred_proba, n_bootstrap=1000, alpha=0.05,
                          seed=42):
    """Bootstrap CI for accuracy (threshold=0.5)."""
    def _acc(y, p):
        return accuracy_score(y, (p >= 0.5).astype(int))
    return bootstrap_ci(y_true, y_pred_proba, _acc,
                        n_bootstrap=n_bootstrap, alpha=alpha, seed=seed)


# 2. Calibration Slope + Intercept ------------------------------------

def calibration_slope_intercept(y_true, y_pred_proba, n_boot=200, seed=42):
    """Cox recalibration: logistic regression of y on logit(p).

    Args:
        y_true: binary array.
        y_pred_proba: predicted probabilities.
        n_boot: bootstrap resamples for CIs.
        seed: random seed.

    Returns:
        dict with slope, intercept, slope_ci, intercept_ci.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    p = np.asarray(y_pred_proba, dtype=np.float64)
    p = np.clip(p, 0.001, 0.999)
    logit_p = np.log(p / (1 - p))

    lr = LogisticRegression(solver="lbfgs", penalty=None, max_iter=1000)
    lr.fit(logit_p.reshape(-1, 1), y_true)
    slope = float(lr.coef_[0][0])
    intercept = float(lr.intercept_[0])

    # Bootstrap CIs
    rng = np.random.RandomState(seed)
    boot_slopes = []
    boot_intercepts = []
    n = len(y_true)
    for _ in range(n_boot):
        idx = rng.choice(n, n, replace=True)
        y_b = y_true[idx]
        lp_b = logit_p[idx]
        if len(np.unique(y_b)) < 2:
            continue
        try:
            lr_b = LogisticRegression(solver="lbfgs", penalty=None,
                                      max_iter=1000)
            lr_b.fit(lp_b.reshape(-1, 1), y_b)
            boot_slopes.append(float(lr_b.coef_[0][0]))
            boot_intercepts.append(float(lr_b.intercept_[0]))
        except Exception:
            continue

    if len(boot_slopes) >= 10:
        slope_ci = (float(np.percentile(boot_slopes, 2.5)),
                    float(np.percentile(boot_slopes, 97.5)))
        intercept_ci = (float(np.percentile(boot_intercepts, 2.5)),
                        float(np.percentile(boot_intercepts, 97.5)))
    else:
        slope_ci = (slope, slope)
        intercept_ci = (intercept, intercept)

    return {"slope": slope, "intercept": intercept,
            "slope_ci": slope_ci, "intercept_ci": intercept_ci}


# 3. Learning Curves --------------------------------------------------

def learning_curves(trials, n_points=5, cv=3, seed=42):
    """Compute learning curves using GradientBoostingClassifier.

    Args:
        trials: list of trial dicts.
        n_points: number of training-size points.
        cv: cross-validation folds.
        seed: random seed.

    Returns:
        dict with train_sizes, train_scores, val_scores (mean per size).
    """
    X, y, _ = _extract_Xy(trials)

    n_samples = len(y)
    min_class = min(int(np.sum(y == 0)), int(np.sum(y == 1)))
    effective_cv = min(cv, min_class)
    effective_cv = max(effective_cv, 2)

    clf = GradientBoostingClassifier(
        n_estimators=50, max_depth=3, learning_rate=0.1, random_state=seed)

    # Determine train sizes -- ensure at least effective_cv * 2 samples
    min_train = max(effective_cv * 2, 4)
    max_train = n_samples - max(int(n_samples / effective_cv), 2)
    max_train = max(max_train, min_train + 1)
    sizes = np.linspace(min_train, max_train, n_points).astype(int)
    sizes = np.unique(np.clip(sizes, min_train, max_train))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        train_sizes, train_sc, val_sc = sklearn_learning_curve(
            clf, X, y,
            train_sizes=sizes,
            cv=effective_cv,
            scoring="accuracy",
            random_state=seed,
            shuffle=True,
        )

    return {
        "train_sizes": train_sizes.tolist(),
        "train_scores": np.mean(train_sc, axis=1).tolist(),
        "val_scores": np.mean(val_sc, axis=1).tolist(),
    }


# 4. DeLong Test for AUC Comparison -----------------------------------

def _compute_midrank(x):
    """Compute midranks for a 1-D array (needed by DeLong)."""
    j = np.argsort(x)
    z = x[j]
    n = len(x)
    rank = np.zeros(n, dtype=np.float64)
    i = 0
    while i < n:
        j_start = i
        while i < n - 1 and z[i] == z[i + 1]:
            i += 1
        midrank = 0.5 * (j_start + i)  # 0-based midrank
        for k in range(j_start, i + 1):
            rank[j[k]] = midrank
        i += 1
    return rank


def _fast_delong(y_true, pred1, pred2):
    """Core DeLong statistic computation.

    Returns:
        (auc1, auc2, var_diff) where var_diff is the variance of AUC1 - AUC2.
    """
    y = np.asarray(y_true, dtype=np.int32)
    p1 = np.asarray(pred1, dtype=np.float64)
    p2 = np.asarray(pred2, dtype=np.float64)

    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]
    m = len(pos_idx)  # number of positives
    n = len(neg_idx)  # number of negatives

    if m == 0 or n == 0:
        raise ValueError("Need both positive and negative cases for DeLong.")

    # Structural components (placement values)
    # For each model: V10[i] = fraction of negatives scored below positive i
    # V01[j] = fraction of positives scored above negative j
    def _placement_values(predictions):
        # V10: for each positive, fraction of negatives with lower score
        pos_pred = predictions[pos_idx]
        neg_pred = predictions[neg_idx]

        v10 = np.zeros(m, dtype=np.float64)
        for i in range(m):
            v10[i] = np.mean(pos_pred[i] > neg_pred) + 0.5 * np.mean(
                pos_pred[i] == neg_pred)

        v01 = np.zeros(n, dtype=np.float64)
        for j in range(n):
            v01[j] = np.mean(neg_pred[j] < pos_pred) + 0.5 * np.mean(
                neg_pred[j] == pos_pred)

        return v10, v01

    v10_1, v01_1 = _placement_values(p1)
    v10_2, v01_2 = _placement_values(p2)

    auc1 = float(np.mean(v10_1))
    auc2 = float(np.mean(v10_2))

    # Covariance of the difference
    # S10 = cov of (V10_1 - V10_2)
    d10 = v10_1 - v10_2  # differences for positives
    d01 = v01_1 - v01_2  # differences for negatives

    s10 = float(np.var(d10, ddof=1)) if m > 1 else 0.0
    s01 = float(np.var(d01, ddof=1)) if n > 1 else 0.0

    var_diff = s10 / m + s01 / n

    return auc1, auc2, var_diff


def delong_test(y_true, pred1, pred2):
    """DeLong test for comparing two correlated AUC-ROCs.

    Args:
        y_true: binary array of true labels.
        pred1: predicted probabilities from model 1.
        pred2: predicted probabilities from model 2.

    Returns:
        dict with auc1, auc2, z_stat, p_value.
    """
    auc1, auc2, var_diff = _fast_delong(y_true, pred1, pred2)

    if var_diff < 1e-15:
        z_stat = 0.0
    else:
        z_stat = (auc1 - auc2) / math.sqrt(var_diff)

    p_value = 2.0 * sp_stats.norm.sf(abs(z_stat))

    return {"auc1": auc1, "auc2": auc2,
            "z_stat": float(z_stat), "p_value": float(p_value)}


# 5. SMOTE ------------------------------------------------------------

def apply_smote(X, y, k=5, seed=42):
    """Synthetic Minority Oversampling Technique.

    Oversamples the minority class to match the majority class size.

    Args:
        X: 2-D feature array (n_samples, n_features).
        y: 1-D binary label array.
        k: number of nearest neighbours.
        seed: random seed.

    Returns:
        (X_augmented, y_augmented) with balanced classes.
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.int32)
    rng = np.random.RandomState(seed)

    classes, counts = np.unique(y, return_counts=True)
    if len(classes) < 2:
        return X.copy(), y.copy()

    majority_class = classes[np.argmax(counts)]
    minority_class = classes[np.argmin(counts)]
    n_majority = int(np.max(counts))
    n_minority = int(np.min(counts))

    minority_idx = np.where(y == minority_class)[0]
    X_min = X[minority_idx]

    n_synthetic = n_majority - n_minority
    if n_synthetic <= 0:
        return X.copy(), y.copy()

    # Effective k: can't exceed minority count - 1
    k_eff = min(k, n_minority - 1)
    if k_eff < 1:
        k_eff = 1

    # Pairwise distances within minority class
    # Use simple Euclidean distance
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=k_eff + 1, metric="euclidean")
    nn.fit(X_min)
    distances, indices = nn.kneighbors(X_min)
    # indices[:, 0] is self, so neighbours are indices[:, 1:]
    neighbor_indices = indices[:, 1:]

    synthetics = []
    for _ in range(n_synthetic):
        # Pick a random minority sample
        i = rng.randint(0, n_minority)
        # Pick a random neighbour
        nn_idx = rng.randint(0, k_eff)
        j = neighbor_indices[i, nn_idx]
        # Interpolate
        lam = rng.rand()
        synthetic = X_min[i] + lam * (X_min[j] - X_min[i])
        synthetics.append(synthetic)

    X_syn = np.array(synthetics)
    y_syn = np.full(n_synthetic, minority_class, dtype=np.int32)

    X_aug = np.vstack([X, X_syn])
    y_aug = np.concatenate([y, y_syn])

    return X_aug, y_aug


# 6. Hyperparameter Tuning --------------------------------------------

def tune_hyperparameters(trials, n_iter=20, cv=3, seed=42):
    """Randomized search over GradientBoostingClassifier hyperparameters.

    Args:
        trials: list of trial dicts.
        n_iter: number of random combinations to try.
        cv: cross-validation folds.
        seed: random seed.

    Returns:
        dict with best_params, best_score, all_results.
    """
    X, y, _ = _extract_Xy(trials)

    min_class = min(int(np.sum(y == 0)), int(np.sum(y == 1)))
    effective_cv = min(cv, min_class)
    effective_cv = max(effective_cv, 2)

    param_dist = {
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "max_depth": [2, 3, 4, 5],
        "n_estimators": [50, 100, 200],
        "min_samples_split": [2, 5, 10],
        "subsample": [0.7, 0.8, 0.9, 1.0],
    }

    clf = GradientBoostingClassifier(random_state=seed)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        search = RandomizedSearchCV(
            clf, param_dist,
            n_iter=n_iter,
            cv=StratifiedKFold(n_splits=effective_cv, shuffle=True,
                               random_state=seed),
            scoring="roc_auc",
            random_state=seed,
            n_jobs=1,
            error_score=0.0,
        )
        search.fit(X, y)

    all_results = []
    for i in range(len(search.cv_results_["params"])):
        all_results.append({
            "params": search.cv_results_["params"][i],
            "mean_score": float(search.cv_results_["mean_test_score"][i]),
        })

    return {
        "best_params": search.best_params_,
        "best_score": float(search.best_score_),
        "all_results": all_results,
    }


# ===================================================================
#  Layer 2 --- Methodologically Novel (5 methods)
# ===================================================================


# 7. Competing Risks Model --------------------------------------------

def competing_risks(trials):
    """Multinomial logistic regression: COMPLETED vs TERMINATED vs WITHDRAWN.

    Args:
        trials: list of trial dicts.

    Returns:
        dict with cause_specific_or and predicted_probabilities.
    """
    X, _, raw_labels = _extract_Xy(trials)
    nct_ids = _get_nct_ids(trials)

    # Build 3-class label
    status_map = {"COMPLETED": 0, "TERMINATED": 1, "WITHDRAWN": 2}
    y_multi = np.array([
        status_map.get(t.get("status", "").upper(), 2) for t in trials
    ], dtype=np.int32)

    class_names = ["COMPLETED", "TERMINATED", "WITHDRAWN"]

    # Fit multinomial logistic regression
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lr = LogisticRegression(
            multi_class="multinomial", solver="lbfgs", max_iter=1000,
            random_state=42, C=1.0)
        lr.fit(X, y_multi)

    # Predicted probabilities for each trial
    probs = lr.predict_proba(X)

    predicted_probabilities = []
    for i, nct_id in enumerate(nct_ids):
        pp = {"nctId": nct_id}
        # Map class indices to probability keys
        for cls_idx, cls_name in enumerate(class_names):
            key = "p_" + cls_name.lower()
            if cls_idx < probs.shape[1]:
                pp[key] = float(probs[i, cls_idx])
            else:
                pp[key] = 0.0
        predicted_probabilities.append(pp)

    # Cause-specific odds ratios (exponentiated coefficients)
    cause_specific_or = []
    for cls_idx in range(lr.coef_.shape[0]):
        cls_name = class_names[cls_idx] if cls_idx < len(class_names) else \
            f"class_{cls_idx}"
        or_vals = np.exp(lr.coef_[cls_idx]).tolist()
        # Confidence intervals via delta method (approximate)
        se = np.sqrt(np.diag(
            np.linalg.inv(
                X.T @ X / len(X) + np.eye(X.shape[1]) * 0.01
            )
        ))
        ci_lower = np.exp(lr.coef_[cls_idx] - 1.96 * se).tolist()
        ci_upper = np.exp(lr.coef_[cls_idx] + 1.96 * se).tolist()

        cause_specific_or.append({
            "cause": cls_name,
            "features": FEATURE_NAMES[:],
            "or_values": or_vals,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
        })

    return {
        "cause_specific_or": cause_specific_or,
        "predicted_probabilities": predicted_probabilities,
    }


# 8. Conformal Prediction ---------------------------------------------

def conformal_prediction(model, trials, alpha=0.10, seed=42):
    """Split conformal prediction for binary completion probability.

    Args:
        model: trained classifier with predict_proba.
        trials: list of trial dicts.
        alpha: miscoverage rate (default 0.10 -> 90 % coverage).
        seed: random seed.

    Returns:
        dict with predictions (list), coverage, mean_width.
    """
    X, y, _ = _extract_Xy(trials)
    nct_ids = _get_nct_ids(trials)
    n = len(y)

    rng = np.random.RandomState(seed)
    indices = rng.permutation(n)

    # Split: 70 % proper training, 30 % calibration
    n_train = max(int(0.7 * n), 2)
    train_idx = indices[:n_train]
    cal_idx = indices[n_train:]

    if len(cal_idx) < 2:
        # Too few calibration samples -- use leave-one-out style
        cal_idx = indices[max(n_train - 2, 0):]
        train_idx = indices[:max(n_train - 2, 0)] if n_train > 2 else indices[:1]

    X_train, y_train = X[train_idx], y[train_idx]
    X_cal, y_cal = X[cal_idx], y[cal_idx]

    # Train a fresh model on proper training set
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        clf = GradientBoostingClassifier(
            n_estimators=100, max_depth=3, learning_rate=0.1,
            random_state=seed)
        # Need both classes
        if len(np.unique(y_train)) < 2:
            clf.fit(X, y)  # fallback to full data
            X_cal, y_cal = X, y
        else:
            clf.fit(X_train, y_train)

    # Compute nonconformity scores on calibration set
    cal_probs = clf.predict_proba(X_cal)
    prob_col = 1 if cal_probs.shape[1] > 1 else 0

    # Nonconformity score: 1 - p_hat(true class)
    scores = np.zeros(len(y_cal), dtype=np.float64)
    for i in range(len(y_cal)):
        if y_cal[i] == 1:
            scores[i] = 1.0 - cal_probs[i, prob_col]
        else:
            scores[i] = cal_probs[i, prob_col]  # 1 - p_hat(class 0) = p_hat(class 1)

    # Quantile threshold
    n_cal = len(scores)
    q_level = math.ceil((1.0 - alpha) * (n_cal + 1)) / n_cal
    q_level = min(q_level, 1.0)
    q_hat = float(np.percentile(scores, q_level * 100))

    # Predict intervals for ALL trials
    all_probs = clf.predict_proba(X)
    predictions = []
    covered = 0
    total_width = 0.0

    for i in range(n):
        p1 = float(all_probs[i, prob_col])
        # Prediction interval: [p1 - q_hat, p1 + q_hat] clipped to [0, 1]
        lower = max(0.0, p1 - q_hat)
        upper = min(1.0, p1 + q_hat)
        width = upper - lower
        total_width += width

        # Check coverage on calibration set members
        if i in set(cal_idx):
            true_val = float(y[i])
            if lower <= true_val <= upper:
                covered += 1

        predictions.append({
            "nctId": nct_ids[i],
            "point": p1,
            "lower": lower,
            "upper": upper,
        })

    n_cal_check = len(cal_idx)
    coverage = covered / n_cal_check if n_cal_check > 0 else 0.0
    mean_width = total_width / n if n > 0 else 0.0

    return {
        "predictions": predictions,
        "coverage": float(coverage),
        "mean_width": float(mean_width),
    }


# 9. Tree SHAP (Simplified) -------------------------------------------

def tree_shap(model, X, feature_names=None):
    """Simplified SHAP values via single-permutation importance per sample.

    For each feature j, shuffle column j and measure the change in
    prediction for each sample.

    Args:
        model: trained classifier with predict_proba.
        X: 2-D feature array (n_samples, n_features).
        feature_names: list of feature names (optional).

    Returns:
        dict with shap_values (2-D array), base_value, feature_names.
    """
    X = np.asarray(X, dtype=np.float64)
    n_samples, n_features = X.shape

    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(n_features)]

    # Base predictions
    probs = model.predict_proba(X)
    prob_col = 1 if probs.shape[1] > 1 else 0
    base_preds = probs[:, prob_col].copy()
    base_value = float(np.mean(base_preds))

    rng = np.random.RandomState(42)
    shap_values = np.zeros((n_samples, n_features), dtype=np.float64)

    for j in range(n_features):
        X_permuted = X.copy()
        perm = rng.permutation(n_samples)
        X_permuted[:, j] = X_permuted[perm, j]

        probs_perm = model.predict_proba(X_permuted)
        perm_preds = probs_perm[:, prob_col]

        shap_values[:, j] = base_preds - perm_preds

    return {
        "shap_values": shap_values,
        "base_value": base_value,
        "feature_names": list(feature_names),
    }


# 10. Bayesian Model Averaging ----------------------------------------

def bayesian_model_averaging(trials, seed=42):
    """BMA over 5 GradientBoosting models with different max_depth.

    Model weights proportional to exp(-BIC/2).

    Args:
        trials: list of trial dicts.
        seed: random seed.

    Returns:
        dict with bma_predictions and model_weights.
    """
    X, y, _ = _extract_Xy(trials)
    nct_ids = _get_nct_ids(trials)
    n = len(y)

    depths = [2, 3, 4, 5, 6]
    models_info = []

    for depth in depths:
        clf = GradientBoostingClassifier(
            n_estimators=100, max_depth=depth, learning_rate=0.1,
            random_state=seed)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            clf.fit(X, y)

        probs = clf.predict_proba(X)
        prob_col = 1 if probs.shape[1] > 1 else 0
        p = probs[:, prob_col]
        p_clipped = np.clip(p, 1e-10, 1.0 - 1e-10)

        # Log-likelihood
        ll = float(np.sum(
            y * np.log(p_clipped) + (1 - y) * np.log(1 - p_clipped)
        ))

        # Number of effective parameters (approximate)
        k_params = depth * 100 + 1  # rough proxy
        bic = -2 * ll + k_params * math.log(n)

        models_info.append({
            "name": f"GBM_depth{depth}",
            "clf": clf,
            "probs": p.copy(),
            "bic": bic,
            "k_params": k_params,
        })

    # Compute BMA weights
    bics = np.array([m["bic"] for m in models_info])
    # Shift for numerical stability
    bic_min = np.min(bics)
    log_weights = -0.5 * (bics - bic_min)
    weights = np.exp(log_weights)
    weights = weights / np.sum(weights)

    # BMA predictions
    bma_probs = np.zeros(n, dtype=np.float64)
    for i, m in enumerate(models_info):
        bma_probs += weights[i] * m["probs"]

    # CIs from weighted model predictions (approximate)
    bma_predictions = []
    for i in range(n):
        model_preds = np.array([m["probs"][i] for m in models_info])
        # Weighted mean
        wmean = float(bma_probs[i])
        # Weighted variance
        wvar = float(np.sum(weights * (model_preds - wmean) ** 2))
        wsd = math.sqrt(max(wvar, 0.0))
        ci_lower = max(0.0, wmean - 1.96 * wsd)
        ci_upper = min(1.0, wmean + 1.96 * wsd)

        bma_predictions.append({
            "nctId": nct_ids[i],
            "prob": wmean,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
        })

    model_weights = []
    for i, m in enumerate(models_info):
        model_weights.append({
            "model_name": m["name"],
            "bic": float(m["bic"]),
            "weight": float(weights[i]),
        })

    return {
        "bma_predictions": bma_predictions,
        "model_weights": model_weights,
    }


# 11. Survival Prediction (Time-to-Completion) ------------------------

def survival_prediction(trials, seed=42):
    """Accelerated failure time model for time-to-completion.

    Uses log-transformed completion months as response. Completed trials
    have observed times; terminated/withdrawn trials are right-censored.

    Args:
        trials: list of trial dicts.
        seed: random seed.

    Returns:
        dict with predictions, rmse, concordance.
    """
    X, _, raw_labels = _extract_Xy(trials)
    nct_ids = _get_nct_ids(trials)
    n = len(trials)

    # Determine observed times and censoring
    times = np.zeros(n, dtype=np.float64)
    censored = np.zeros(n, dtype=bool)

    for i, label in enumerate(raw_labels):
        mtc = label.get("months_to_completion")
        if mtc is not None and mtc > 0:
            times[i] = mtc
            censored[i] = (label["completed"] == 0)
        else:
            # Impute time for missing: use median of observed completed times
            censored[i] = True
            times[i] = 0  # will fill later

    # Fill missing times with median of observed
    observed_mask = times > 0
    if np.any(observed_mask):
        median_time = float(np.median(times[observed_mask]))
    else:
        median_time = 24.0  # fallback

    for i in range(n):
        if times[i] <= 0:
            times[i] = median_time

    # Train on completed (uncensored) trials only
    completed_mask = ~censored
    n_completed = int(np.sum(completed_mask))

    if n_completed < 3:
        # Not enough completed trials for meaningful regression
        predictions = []
        for i in range(n):
            predictions.append({
                "nctId": nct_ids[i],
                "predicted_months": float(median_time),
                "actual_months": float(times[i]),
                "censored": bool(censored[i]),
            })
        return {"predictions": predictions, "rmse": 0.0, "concordance": 0.0}

    X_train = X[completed_mask]
    log_t_train = np.log(times[completed_mask])

    ridge = Ridge(alpha=1.0, random_state=seed)
    ridge.fit(X_train, log_t_train)

    # Predict for all trials
    log_t_pred = ridge.predict(X)
    t_pred = np.exp(log_t_pred)
    # Ensure positive
    t_pred = np.maximum(t_pred, 0.1)

    predictions = []
    for i in range(n):
        predictions.append({
            "nctId": nct_ids[i],
            "predicted_months": float(round(t_pred[i], 2)),
            "actual_months": float(round(times[i], 2)),
            "censored": bool(censored[i]),
        })

    # RMSE on completed trials only
    residuals = log_t_train - ridge.predict(X_train)
    rmse = float(math.sqrt(np.mean(residuals ** 2)))

    # Concordance index on completed trials
    concordance = _concordance_index(times[completed_mask], t_pred[completed_mask])

    return {
        "predictions": predictions,
        "rmse": rmse,
        "concordance": float(concordance),
    }


def _concordance_index(y_true, y_pred):
    """Compute Harrell's concordance index.

    C-index = proportion of concordant pairs among all comparable pairs.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n = len(y_true)
    concordant = 0
    discordant = 0
    tied = 0

    for i in range(n):
        for j in range(i + 1, n):
            if y_true[i] == y_true[j]:
                continue  # not comparable
            if y_true[i] < y_true[j]:
                if y_pred[i] < y_pred[j]:
                    concordant += 1
                elif y_pred[i] > y_pred[j]:
                    discordant += 1
                else:
                    tied += 1
            else:
                if y_pred[i] > y_pred[j]:
                    concordant += 1
                elif y_pred[i] < y_pred[j]:
                    discordant += 1
                else:
                    tied += 1

    total = concordant + discordant + tied
    if total == 0:
        return 0.5
    return (concordant + 0.5 * tied) / total
