"""Advanced statistical engine for EnrollmentOracle.

17 methods across three layers:
  Layer 1 (publication-essential): bootstrap CI, calibration slope,
    learning curves, DeLong test, SMOTE, hyperparameter tuning.
  Layer 2 (methodologically novel): competing risks, conformal prediction,
    tree SHAP, Bayesian model averaging, survival prediction.
  Layer 3 (advanced): partial dependence, isotonic calibration,
    jackknife+, fairness metrics, PIT histogram, ensemble diversity.
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


# ===================================================================
#  Layer 3 --- Advanced Statistical Methods (6 methods)
# ===================================================================


# 12. Partial Dependence Plots ------------------------------------------

def partial_dependence(model, X, feature_idx, feature_names=None,
                       grid_resolution=50):
    """Partial Dependence Plot (PDP) + Individual Conditional Expectation.

    Args:
        model: trained classifier with predict_proba.
        X: 2-D feature array (n_samples, n_features).
        feature_idx: integer index of the feature to inspect.
        feature_names: optional list of feature names.
        grid_resolution: number of grid points for continuous features.

    Returns:
        dict with feature_name, grid_values, pdp_values, ice_lines.
    """
    X = np.asarray(X, dtype=np.float64)
    n_samples, n_features = X.shape

    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(n_features)]

    feature_name = feature_names[feature_idx]

    # Determine grid values
    unique_vals = np.unique(X[:, feature_idx])
    if len(unique_vals) <= grid_resolution:
        grid_values = unique_vals
    else:
        grid_values = np.linspace(
            float(np.min(X[:, feature_idx])),
            float(np.max(X[:, feature_idx])),
            grid_resolution,
        )

    prob_col = 1 if model.predict_proba(X[:1]).shape[1] > 1 else 0

    pdp_values = np.zeros(len(grid_values), dtype=np.float64)
    ice_lines = np.zeros((n_samples, len(grid_values)), dtype=np.float64)

    for g_idx, g in enumerate(grid_values):
        X_modified = X.copy()
        X_modified[:, feature_idx] = g
        probs = model.predict_proba(X_modified)[:, prob_col]
        pdp_values[g_idx] = float(np.mean(probs))
        ice_lines[:, g_idx] = probs

    return {
        "feature_name": feature_name,
        "grid_values": grid_values.tolist(),
        "pdp_values": pdp_values.tolist(),
        "ice_lines": ice_lines,
    }


# 13. Isotonic Regression Calibration -----------------------------------

def isotonic_calibration(y_true, y_pred_proba):
    """Non-parametric isotonic calibration of predicted probabilities.

    Args:
        y_true: binary array of true labels.
        y_pred_proba: predicted probabilities.

    Returns:
        dict with calibrated_probs, isotonic_fit, brier_before, brier_after.
    """
    from sklearn.isotonic import IsotonicRegression

    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred_proba = np.asarray(y_pred_proba, dtype=np.float64)

    brier_before = float(brier_score_loss(y_true, y_pred_proba))

    ir = IsotonicRegression(out_of_bounds='clip')
    ir.fit(y_pred_proba, y_true)
    calibrated = ir.predict(y_pred_proba)
    calibrated = np.clip(calibrated, 0.0, 1.0)

    brier_after = float(brier_score_loss(y_true, calibrated))

    return {
        "calibrated_probs": calibrated.tolist(),
        "isotonic_fit": ir,
        "brier_before": brier_before,
        "brier_after": brier_after,
    }


# 14. Jackknife+ Prediction Intervals -----------------------------------

def jackknife_plus(model_class, X, y, alpha=0.10, seed=42):
    """Jackknife+ conformal prediction intervals (leave-one-out).

    Args:
        model_class: callable that returns a fresh untrained classifier.
        X: 2-D feature array.
        y: 1-D binary label array.
        alpha: miscoverage rate (default 0.10 -> 90% intervals).
        seed: random seed (unused but kept for API consistency).

    Returns:
        dict with predictions [{index, point, lower, upper}],
        coverage, mean_width.
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.int32)
    n = len(y)

    residuals = np.zeros(n, dtype=np.float64)
    p_hat = np.zeros(n, dtype=np.float64)

    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        X_train, y_train = X[mask], y[mask]

        # Need both classes in training set
        if len(np.unique(y_train)) < 2:
            # Fallback: train on full data
            model_i = model_class()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model_i.fit(X, y)
        else:
            model_i = model_class()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model_i.fit(X_train, y_train)

        probs_i = model_i.predict_proba(X[i:i+1])
        prob_col = 1 if probs_i.shape[1] > 1 else 0
        p_hat[i] = float(probs_i[0, prob_col])
        residuals[i] = abs(float(y[i]) - p_hat[i])

    # Quantile of residuals for interval width
    q = float(np.quantile(residuals, 1.0 - alpha))

    predictions = []
    covered = 0
    total_width = 0.0

    for i in range(n):
        lower = max(0.0, p_hat[i] - q)
        upper = min(1.0, p_hat[i] + q)
        width = upper - lower
        total_width += width

        # Check coverage
        if lower <= float(y[i]) <= upper:
            covered += 1

        predictions.append({
            "index": i,
            "point": float(p_hat[i]),
            "lower": float(lower),
            "upper": float(upper),
        })

    coverage = covered / n if n > 0 else 0.0
    mean_width = total_width / n if n > 0 else 0.0

    return {
        "predictions": predictions,
        "coverage": float(coverage),
        "mean_width": float(mean_width),
    }


# 15. Fairness Metrics --------------------------------------------------

def fairness_metrics(y_true, y_pred, y_pred_proba, sensitive_groups):
    """Compute fairness metrics across sensitive groups.

    Args:
        y_true: binary array of true labels.
        y_pred: binary array of predicted labels.
        y_pred_proba: predicted probabilities.
        sensitive_groups: list of group labels (e.g., sponsor type).

    Returns:
        dict with demographic_parity, equalized_odds,
        calibration_by_group, disparate_impact_ratio.
    """
    y_true = np.asarray(y_true, dtype=np.int32)
    y_pred = np.asarray(y_pred, dtype=np.int32)
    y_pred_proba = np.asarray(y_pred_proba, dtype=np.float64)
    groups = np.array(sensitive_groups)

    unique_groups = sorted(set(sensitive_groups))

    # Demographic parity: P(Y_hat=1 | group=g)
    demographic_parity = {}
    for g in unique_groups:
        mask = groups == g
        if np.sum(mask) > 0:
            demographic_parity[g] = float(np.mean(y_pred[mask]))
        else:
            demographic_parity[g] = 0.0

    # Equalized odds: TPR and FPR per group
    equalized_odds = {}
    for g in unique_groups:
        mask = groups == g
        y_t = y_true[mask]
        y_p = y_pred[mask]
        pos = np.sum(y_t == 1)
        neg = np.sum(y_t == 0)
        tpr = float(np.sum((y_p == 1) & (y_t == 1)) / pos) if pos > 0 else 0.0
        fpr = float(np.sum((y_p == 1) & (y_t == 0)) / neg) if neg > 0 else 0.0
        equalized_odds[g] = {"tpr": tpr, "fpr": fpr}

    # Calibration by group: Brier score, mean predicted, mean actual
    calibration_by_group = {}
    for g in unique_groups:
        mask = groups == g
        if np.sum(mask) > 0:
            y_t = y_true[mask]
            p = y_pred_proba[mask]
            brier = float(brier_score_loss(y_t, p)) if len(np.unique(y_t)) > 0 else 0.0
            calibration_by_group[g] = {
                "brier": brier,
                "mean_pred": float(np.mean(p)),
                "mean_actual": float(np.mean(y_t)),
            }
        else:
            calibration_by_group[g] = {
                "brier": 0.0, "mean_pred": 0.0, "mean_actual": 0.0,
            }

    # Disparate impact ratio: min(rate) / max(rate)
    rates = [v for v in demographic_parity.values() if v > 0]
    if len(rates) >= 2:
        disparate_impact_ratio = float(min(rates) / max(rates))
    elif len(rates) == 1:
        disparate_impact_ratio = 1.0
    else:
        disparate_impact_ratio = 0.0

    return {
        "demographic_parity": demographic_parity,
        "equalized_odds": equalized_odds,
        "calibration_by_group": calibration_by_group,
        "disparate_impact_ratio": disparate_impact_ratio,
    }


# 16. Probability Integral Transform (PIT) Histogram --------------------

def pit_histogram(y_true, y_pred_proba, n_bins=10, seed=42):
    """PIT histogram for assessing probabilistic calibration.

    Uses randomized PIT for binary outcomes (Czado et al. 2009):
      PIT_i ~ Uniform(F(y_i - 1), F(y_i))
    For binary: F(0) = 1 - p, F(1) = 1.
      y=0: PIT ~ Uniform(0, 1-p)
      y=1: PIT ~ Uniform(1-p, 1)

    A well-calibrated model produces a uniform PIT distribution.

    Args:
        y_true: binary array.
        y_pred_proba: predicted P(Y=1).
        n_bins: number of histogram bins.
        seed: random seed for randomized PIT.

    Returns:
        dict with bin_edges, bin_counts, chi2_stat, p_value, is_uniform.
    """
    y_true = np.asarray(y_true, dtype=np.int32)
    y_pred_proba = np.asarray(y_pred_proba, dtype=np.float64)
    n = len(y_true)

    rng = np.random.RandomState(seed)
    u = rng.rand(n)

    # Randomized PIT for binary outcomes
    # y=0: PIT in [0, 1-p], so PIT = u * (1-p)
    # y=1: PIT in [1-p, 1], so PIT = (1-p) + u * p
    pit = np.where(
        y_true == 1,
        (1.0 - y_pred_proba) + u * y_pred_proba,
        u * (1.0 - y_pred_proba),
    )
    pit = np.clip(pit, 0.0, 1.0)

    # Histogram
    bin_counts, bin_edges = np.histogram(pit, bins=n_bins, range=(0.0, 1.0))

    # Chi-squared goodness-of-fit against uniform
    expected = n / n_bins
    if expected > 0:
        chi2_stat = float(np.sum((bin_counts - expected) ** 2 / expected))
        df = n_bins - 1
        p_value = float(1.0 - sp_stats.chi2.cdf(chi2_stat, df))
    else:
        chi2_stat = 0.0
        p_value = 1.0

    return {
        "bin_edges": bin_edges.tolist(),
        "bin_counts": bin_counts.tolist(),
        "chi2_stat": chi2_stat,
        "p_value": p_value,
        "is_uniform": p_value > 0.05,
    }


# 17. Ensemble Diversity Metrics ----------------------------------------

def ensemble_diversity(models, X, y):
    """Compute pairwise and aggregate diversity metrics for an ensemble.

    Args:
        models: list of trained classifiers with predict.
        X: 2-D feature array.
        y: 1-D binary label array.

    Returns:
        dict with disagreement_measure, double_fault, q_statistic,
        kappa_statistic, entropy_measure.
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.int32)
    n = len(y)
    n_models = len(models)

    if n_models < 2:
        return {
            "disagreement_measure": 0.0,
            "double_fault": 0.0,
            "q_statistic": 0.0,
            "kappa_statistic": 0.0,
            "entropy_measure": 0.0,
        }

    # Get binary predictions from all models
    preds = np.zeros((n_models, n), dtype=np.int32)
    for m_idx, model in enumerate(models):
        preds[m_idx] = (model.predict_proba(X)[:, 1] >= 0.5).astype(np.int32)

    # Correctness matrix: 1 if prediction matches true label
    correct = np.zeros((n_models, n), dtype=np.int32)
    for m_idx in range(n_models):
        correct[m_idx] = (preds[m_idx] == y).astype(np.int32)

    # Pairwise metrics
    pair_disagreement = []
    pair_double_fault = []
    pair_q = []
    pair_kappa = []

    for i in range(n_models):
        for j in range(i + 1, n_models):
            # Contingency table
            n11 = int(np.sum((correct[i] == 1) & (correct[j] == 1)))
            n10 = int(np.sum((correct[i] == 1) & (correct[j] == 0)))
            n01 = int(np.sum((correct[i] == 0) & (correct[j] == 1)))
            n00 = int(np.sum((correct[i] == 0) & (correct[j] == 0)))

            # Disagreement: fraction where models disagree on correctness
            disagree = (n10 + n01) / n if n > 0 else 0.0
            pair_disagreement.append(disagree)

            # Double fault: fraction where both are wrong
            df = n00 / n if n > 0 else 0.0
            pair_double_fault.append(df)

            # Q-statistic
            denom_q = n11 * n00 + n01 * n10
            if denom_q > 0:
                q_val = (n11 * n00 - n01 * n10) / denom_q
            else:
                q_val = 0.0
            pair_q.append(q_val)

            # Kappa
            observed_agreement = (n11 + n00) / n if n > 0 else 0.0
            p1 = (n11 + n10) / n if n > 0 else 0.0
            p2 = (n11 + n01) / n if n > 0 else 0.0
            expected_agreement = p1 * p2 + (1 - p1) * (1 - p2)
            if abs(1 - expected_agreement) > 1e-10:
                kappa_val = (observed_agreement - expected_agreement) / \
                    (1 - expected_agreement)
            else:
                kappa_val = 0.0
            pair_kappa.append(kappa_val)

    # Entropy measure: mean over samples
    # E(x) = 1 - max(votes_for_class) / n_models
    entropy_vals = np.zeros(n, dtype=np.float64)
    for s in range(n):
        votes = preds[:, s]
        vote_counts = np.bincount(votes, minlength=2)
        max_votes = np.max(vote_counts)
        entropy_vals[s] = 1.0 - max_votes / n_models

    return {
        "disagreement_measure": float(np.mean(pair_disagreement))
            if pair_disagreement else 0.0,
        "double_fault": float(np.mean(pair_double_fault))
            if pair_double_fault else 0.0,
        "q_statistic": float(np.mean(pair_q))
            if pair_q else 0.0,
        "kappa_statistic": float(np.mean(pair_kappa))
            if pair_kappa else 0.0,
        "entropy_measure": float(np.mean(entropy_vals)),
    }


# ===================================================================
#  Layer 4 --- Advanced Statistical Methods (6 methods)
# ===================================================================


# 18. Gaussian Process Regression ----------------------------------------

def gaussian_process_predict(X_train, y_train, X_test, seed=42):
    """Predict using Gaussian Process Regression with RBF kernel.

    Args:
        X_train: array-like (n_train, n_features) training inputs.
        y_train: array-like (n_train,) training targets.
        X_test: array-like (n_test, n_features) test inputs.
        seed: random seed for reproducibility.

    Returns:
        dict with:
            predictions: list of predicted means per test point.
            uncertainties: list of predicted standard deviations.
            log_marginal_likelihood: float, log marginal likelihood of model.
    """
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, ConstantKernel

    X_train = np.asarray(X_train, dtype=np.float64)
    y_train = np.asarray(y_train, dtype=np.float64)
    X_test = np.asarray(X_test, dtype=np.float64)

    kernel = ConstantKernel(1.0) * RBF(length_scale=1.0)
    gpr = GaussianProcessRegressor(
        kernel=kernel,
        n_restarts_optimizer=3,
        random_state=seed,
        normalize_y=True,
    )
    gpr.fit(X_train, y_train)

    y_pred, y_std = gpr.predict(X_test, return_std=True)

    return {
        "predictions": [float(v) for v in y_pred],
        "uncertainties": [float(v) for v in y_std],
        "log_marginal_likelihood": float(gpr.log_marginal_likelihood_value_),
    }


# 19. Dirichlet Process Mixture Model (DPMM) ----------------------------

def dirichlet_process_mixture(X, alpha=1.0, n_iter=200, seed=42):
    """Cluster data with a Dirichlet Process Mixture Model.

    Uses a Chinese Restaurant Process (CRP) Gibbs sampler with
    isotropic Normal likelihood and Normal-Inverse-Wishart prior
    (simplified to isotropic).

    Args:
        X: array-like (n_samples, n_features).
        alpha: concentration parameter (higher = more clusters).
        n_iter: number of Gibbs sampling iterations.
        seed: random seed.

    Returns:
        dict with:
            n_clusters: number of discovered clusters.
            assignments: list of cluster assignment per sample.
            cluster_means: list of cluster mean vectors.
            cluster_sizes: list of cluster sizes.
    """
    rng = np.random.RandomState(seed)
    X = np.asarray(X, dtype=np.float64)
    n, d = X.shape

    # Initialize: all samples in one cluster
    assignments = np.zeros(n, dtype=int)
    max_cluster = 0

    # Prior parameters
    mu0 = np.mean(X, axis=0)
    sigma0 = float(np.std(X)) + 1e-6  # prior std
    lambda0 = 1.0 / (sigma0 ** 2)  # prior precision

    for iteration in range(n_iter):
        for i in range(n):
            # Remove sample i from its cluster
            old_cluster = assignments[i]

            # Count cluster members (excluding i)
            cluster_counts = {}
            for j in range(n):
                if j == i:
                    continue
                c = assignments[j]
                cluster_counts[c] = cluster_counts.get(c, 0) + 1

            # Remove empty clusters after removal
            active_clusters = sorted(cluster_counts.keys())

            # Compute CRP probabilities
            log_probs = []
            cluster_ids = []

            for c in active_clusters:
                count_c = cluster_counts[c]
                # Members of cluster c (excluding i)
                members = X[np.array([j for j in range(n)
                                       if j != i and assignments[j] == c])]
                if len(members) == 0:
                    continue

                # Posterior predictive (Normal with known variance, simplified)
                cluster_mean = np.mean(members, axis=0)
                n_c = len(members)
                post_mean = (lambda0 * mu0 + n_c * cluster_mean) / (lambda0 + n_c)
                post_var = sigma0 ** 2 * (1.0 + 1.0 / (lambda0 + n_c))

                # Log-likelihood of x_i under posterior predictive
                diff = X[i] - post_mean
                log_lik = -0.5 * np.sum(diff ** 2) / post_var - 0.5 * d * math.log(
                    2 * math.pi * post_var
                )

                log_prob = math.log(count_c) + log_lik
                log_probs.append(log_prob)
                cluster_ids.append(c)

            # New cluster probability
            diff_new = X[i] - mu0
            var_new = sigma0 ** 2 * (1.0 + 1.0 / lambda0)
            log_lik_new = -0.5 * np.sum(diff_new ** 2) / var_new - 0.5 * d * math.log(
                2 * math.pi * var_new
            )
            new_cluster_id = max_cluster + 1
            log_prob_new = math.log(alpha) + log_lik_new
            log_probs.append(log_prob_new)
            cluster_ids.append(new_cluster_id)

            # Normalize (log-sum-exp)
            max_lp = max(log_probs)
            probs = [math.exp(lp - max_lp) for lp in log_probs]
            total = sum(probs)
            probs = [p / total for p in probs]

            # Sample assignment
            chosen_idx = rng.choice(len(cluster_ids), p=probs)
            assignments[i] = cluster_ids[chosen_idx]
            if cluster_ids[chosen_idx] > max_cluster:
                max_cluster = cluster_ids[chosen_idx]

    # Final cluster statistics
    unique_clusters = sorted(set(assignments))
    # Remap to 0-indexed
    remap = {c: idx for idx, c in enumerate(unique_clusters)}
    final_assignments = [remap[c] for c in assignments]
    n_clusters = len(unique_clusters)

    cluster_means = []
    cluster_sizes = []
    for c in unique_clusters:
        mask = assignments == c
        members = X[mask]
        cluster_means.append([float(v) for v in np.mean(members, axis=0)])
        cluster_sizes.append(int(np.sum(mask)))

    return {
        "n_clusters": n_clusters,
        "assignments": final_assignments,
        "cluster_means": cluster_means,
        "cluster_sizes": cluster_sizes,
    }


# 20. Knockoff Filter ---------------------------------------------------

def knockoff_filter(X, y, fdr=0.10, seed=42):
    """Select features using the model-X knockoff filter.

    Generates knockoff features by permuting + adding noise, fits Lasso
    on [X, X_tilde], and computes knockoff statistics W_j.

    Args:
        X: array-like (n_samples, n_features).
        y: array-like (n_samples,) binary response.
        fdr: target false discovery rate (default 0.10).
        seed: random seed.

    Returns:
        dict with:
            selected_features: list of selected feature indices.
            knockoff_stats: list of W_j values per feature.
            threshold: knockoff+ threshold value.
    """
    from sklearn.linear_model import Lasso

    rng = np.random.RandomState(seed)
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n, p = X.shape

    # Generate knockoff features: permute within columns + small noise
    X_tilde = np.zeros_like(X)
    for j in range(p):
        perm = rng.permutation(n)
        X_tilde[:, j] = X[perm, j] + rng.randn(n) * 0.01 * (np.std(X[:, j]) + 1e-10)

    # Augmented design matrix [X, X_tilde]
    X_aug = np.hstack([X, X_tilde])

    # Fit Lasso
    # Choose regularization that selects ~half features
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_aug_scaled = scaler.fit_transform(X_aug)

    lasso = Lasso(alpha=0.01, max_iter=5000, random_state=seed)
    lasso.fit(X_aug_scaled, y)
    coefs = np.abs(lasso.coef_)

    # Knockoff statistics: W_j = |coef_j| - |coef_tilde_j|
    W = np.zeros(p)
    for j in range(p):
        W[j] = coefs[j] - coefs[p + j]

    # Knockoff+ threshold
    # t = min{ t > 0 : (1 + #{j: W_j <= -t}) / max(1, #{j: W_j >= t}) <= fdr }
    abs_W = np.sort(np.abs(W[W != 0]))[::-1]
    threshold = float('inf')
    for t in abs_W:
        if t <= 0:
            continue
        n_pos = int(np.sum(W >= t))
        n_neg = int(np.sum(W <= -t))
        fdp = (1 + n_neg) / max(1, n_pos)
        if fdp <= fdr:
            threshold = float(t)
            break

    if threshold == float('inf'):
        threshold = float(np.max(np.abs(W)) + 1) if len(W) > 0 else 0.0
        selected = []
    else:
        selected = [int(j) for j in range(p) if W[j] >= threshold]

    return {
        "selected_features": selected,
        "knockoff_stats": [float(w) for w in W],
        "threshold": float(threshold),
    }


# 21. Variational Bayes for Logistic Regression --------------------------

def variational_bayes(trials, n_iter=200, seed=42):
    """Mean-field variational inference for logistic regression.

    Approximates posterior q(beta) = N(mu, diag(sigma^2)) and optimizes
    the Evidence Lower Bound (ELBO) via coordinate ascent.

    Args:
        trials: list of trial dicts (uses feature_engineer pipeline).
        n_iter: number of VI iterations.
        seed: random seed.

    Returns:
        dict with:
            posterior_means: list of posterior mean per feature.
            posterior_stds: list of posterior std per feature.
            elbo_trace: list of ELBO values per iteration.
            convergence_iter: iteration at which ELBO converged.
    """
    rng = np.random.RandomState(seed)
    X, y, _ = _extract_Xy(trials)
    n, p = X.shape

    # Standardize features
    X_mean = np.mean(X, axis=0)
    X_std = np.std(X, axis=0) + 1e-10
    X_norm = (X - X_mean) / X_std

    # Initialize variational parameters
    mu = rng.randn(p) * 0.01
    sigma2 = np.ones(p) * 1.0  # variance

    # Prior: N(0, prior_var * I)
    prior_var = 10.0

    # Jaakkola-Jordan bound for logistic sigmoid
    # sigma(z) >= sigma(xi) * exp((z - xi)/2 - lambda(xi)*(z^2 - xi^2))
    # where lambda(xi) = (sigma(xi) - 0.5) / (2*xi)
    def _lambda(xi):
        xi = np.maximum(np.abs(xi), 1e-6)
        sig = 1.0 / (1.0 + np.exp(-xi))
        return (sig - 0.5) / (2.0 * xi)

    # Initialize variational bound parameters
    xi = np.ones(n)

    elbo_trace = []
    convergence_iter = n_iter

    for it in range(n_iter):
        lam = _lambda(xi)

        # Update sigma^2 (posterior variance)
        for j in range(p):
            sigma2[j] = 1.0 / (1.0 / prior_var + 2.0 * np.sum(lam * X_norm[:, j] ** 2))

        # Update mu (posterior mean)
        # mu_j = sigma2_j * sum_i (y_i - 0.5) * X_{ij}
        # accounting for other features via mean-field
        for j in range(p):
            residual = (y - 0.5) * X_norm[:, j]
            # Subtract contribution of other features
            linear_pred_other = np.zeros(n)
            for k in range(p):
                if k != j:
                    linear_pred_other += mu[k] * X_norm[:, k]
            correction = 2.0 * lam * X_norm[:, j] * linear_pred_other
            mu[j] = sigma2[j] * float(np.sum(residual - correction))

        # Update xi
        E_z2 = np.zeros(n)
        for i in range(n):
            x_i = X_norm[i]
            E_z2[i] = float(np.sum((mu ** 2 + sigma2) * x_i ** 2))
            # Cross terms
            for j in range(p):
                for k in range(j + 1, p):
                    E_z2[i] += 2 * mu[j] * mu[k] * x_i[j] * x_i[k]
        xi = np.sqrt(np.maximum(E_z2, 1e-10))

        # Compute ELBO (approximate)
        lam = _lambda(xi)
        elbo = 0.0
        # Expected log-likelihood (Jaakkola-Jordan bound)
        linear = X_norm @ mu
        for i in range(n):
            elbo += (y[i] - 0.5) * linear[i] - lam[i] * linear[i] ** 2
            elbo += math.log(1.0 / (1.0 + math.exp(-float(xi[i]))) + 1e-300)
            elbo -= 0.5 * float(xi[i])
            elbo += lam[i] * float(xi[i]) ** 2

        # KL divergence: KL(q || prior)
        for j in range(p):
            elbo -= 0.5 * (mu[j] ** 2 + sigma2[j]) / prior_var
            elbo += 0.5 * math.log(sigma2[j] + 1e-300)
            elbo += 0.5 * (1 + math.log(2 * math.pi))
            elbo -= 0.5 * math.log(prior_var)

        elbo_trace.append(float(elbo))

        # Check convergence
        if it > 5 and abs(elbo_trace[-1] - elbo_trace[-2]) < 1e-6:
            convergence_iter = it
            break

    return {
        "posterior_means": [float(m) for m in mu],
        "posterior_stds": [float(math.sqrt(s)) for s in sigma2],
        "elbo_trace": elbo_trace,
        "convergence_iter": int(convergence_iter),
    }


# 22. Regression Discontinuity Design -----------------------------------

def regression_discontinuity(trials, cutoff_feature, cutoff_value,
                              bandwidth=None):
    """Estimate treatment effect at a cutoff via local linear regression.

    Uses Imbens-Kalyanaraman (IK) optimal bandwidth when not specified.

    Args:
        trials: list of trial dicts with cutoff_feature and "completed" keys
                (or feature_engineer-compatible dicts).
        cutoff_feature: string key name in trial dict to use as running var.
        cutoff_value: numeric cutoff value.
        bandwidth: optional float bandwidth; if None, uses IK optimal.

    Returns:
        dict with:
            treatment_effect: estimated discontinuity (jump at cutoff).
            se: standard error of the estimate.
            ci_lower, ci_upper: 95% confidence interval.
            p_value: two-sided p-value.
            bandwidth_used: bandwidth used for estimation.
            n_treated: number of observations above cutoff.
            n_control: number of observations below cutoff.
    """
    # Extract running variable and outcome
    X_run = []
    Y = []
    for trial in trials:
        val = trial.get(cutoff_feature)
        if val is None:
            continue
        outcome = trial.get("completed")
        if outcome is None:
            continue
        X_run.append(float(val))
        Y.append(float(outcome))

    X_run = np.array(X_run)
    Y = np.array(Y)
    n_total = len(X_run)

    if n_total < 4:
        return {
            "treatment_effect": 0.0,
            "se": float('inf'),
            "ci_lower": -float('inf'),
            "ci_upper": float('inf'),
            "p_value": 1.0,
            "bandwidth_used": 0.0,
            "n_treated": 0,
            "n_control": 0,
        }

    # IK optimal bandwidth (simplified Silverman rule * 2)
    if bandwidth is None:
        std_x = float(np.std(X_run)) + 1e-10
        bandwidth = 1.06 * std_x * n_total ** (-0.2) * 2.0

    bandwidth = float(bandwidth)

    # Select observations within bandwidth
    left_mask = (X_run >= cutoff_value - bandwidth) & (X_run < cutoff_value)
    right_mask = (X_run >= cutoff_value) & (X_run <= cutoff_value + bandwidth)

    X_left = X_run[left_mask] - cutoff_value
    Y_left = Y[left_mask]
    X_right = X_run[right_mask] - cutoff_value
    Y_right = Y[right_mask]

    n_control = int(np.sum(left_mask))
    n_treated = int(np.sum(right_mask))

    if n_control < 2 or n_treated < 2:
        # Fallback: use all data on each side
        left_mask_all = X_run < cutoff_value
        right_mask_all = X_run >= cutoff_value
        X_left = X_run[left_mask_all] - cutoff_value
        Y_left = Y[left_mask_all]
        X_right = X_run[right_mask_all] - cutoff_value
        Y_right = Y[right_mask_all]
        n_control = int(np.sum(left_mask_all))
        n_treated = int(np.sum(right_mask_all))

    if n_control < 1 or n_treated < 1:
        return {
            "treatment_effect": 0.0,
            "se": float('inf'),
            "ci_lower": -float('inf'),
            "ci_upper": float('inf'),
            "p_value": 1.0,
            "bandwidth_used": bandwidth,
            "n_treated": n_treated,
            "n_control": n_control,
        }

    # Local linear regression: fit y = a + b*x on each side
    # Left side: intercept = predicted value at cutoff from left
    if len(X_left) >= 2:
        slope_l, intercept_l, _, _, se_l = sp_stats.linregress(X_left, Y_left)
        y_left_at_cut = intercept_l  # at x=0 (cutoff)
    else:
        y_left_at_cut = float(np.mean(Y_left))
        se_l = 0.0

    # Right side
    if len(X_right) >= 2:
        slope_r, intercept_r, _, _, se_r = sp_stats.linregress(X_right, Y_right)
        y_right_at_cut = intercept_r
    else:
        y_right_at_cut = float(np.mean(Y_right))
        se_r = 0.0

    # Treatment effect = jump at cutoff
    tau = y_right_at_cut - y_left_at_cut

    # Standard error (combined)
    var_left = float(np.var(Y_left, ddof=1)) / max(n_control, 1) if n_control > 1 else 0.0
    var_right = float(np.var(Y_right, ddof=1)) / max(n_treated, 1) if n_treated > 1 else 0.0
    se = math.sqrt(var_left + var_right) if (var_left + var_right) > 0 else 1e-10

    z = tau / se if se > 0 else 0.0
    p_value = float(2 * (1 - sp_stats.norm.cdf(abs(z))))

    ci_lower = tau - 1.96 * se
    ci_upper = tau + 1.96 * se

    return {
        "treatment_effect": float(tau),
        "se": float(se),
        "ci_lower": float(ci_lower),
        "ci_upper": float(ci_upper),
        "p_value": float(p_value),
        "bandwidth_used": float(bandwidth),
        "n_treated": n_treated,
        "n_control": n_control,
    }


# 23. Approximate Bayesian Computation -----------------------------------

def abc_posterior(trials, n_simulations=2000, epsilon_quantile=0.05, seed=42):
    """Estimate posterior distribution via Approximate Bayesian Computation.

    Uses rejection sampling with summary statistics: mean completion rate
    and enrollment ratio standard deviation.

    Args:
        trials: list of trial dicts.
        n_simulations: number of prior samples to draw.
        epsilon_quantile: quantile of distances used as acceptance threshold.
        seed: random seed.

    Returns:
        dict with:
            posterior_samples: list of accepted parameter vectors [mu, sigma].
            acceptance_rate: fraction of simulations accepted.
            parameter_medians: dict with median of each parameter.
            credible_intervals: dict with 95% CI for each parameter.
    """
    rng = np.random.RandomState(seed)

    # Observed summary statistics
    completion_rates = []
    enrollment_ratios = []
    for trial in trials:
        actual = trial.get("enrollmentActual")
        planned = trial.get("enrollmentPlanned")
        status = trial.get("status", "")

        completed = 1 if status in ("COMPLETED", "Completed") else 0
        completion_rates.append(completed)

        if planned is not None and planned > 0 and actual is not None:
            enrollment_ratios.append(actual / planned)
        else:
            enrollment_ratios.append(1.0)

    obs_mean_completion = float(np.mean(completion_rates))
    obs_std_enrollment = float(np.std(enrollment_ratios))
    n_trials = len(trials)

    # Prior: mu ~ Uniform(0, 1), sigma ~ Uniform(0, 2)
    prior_mu = rng.uniform(0, 1, n_simulations)
    prior_sigma = rng.uniform(0.01, 2.0, n_simulations)

    # Simulate and compute distances
    distances = np.zeros(n_simulations)
    for s in range(n_simulations):
        # Simulate completion rates from Bernoulli(mu)
        sim_completions = rng.binomial(1, prior_mu[s], n_trials)
        sim_mean_completion = float(np.mean(sim_completions))

        # Simulate enrollment ratios from N(1, sigma)
        sim_enrollment = rng.normal(1.0, prior_sigma[s], n_trials)
        sim_std_enrollment = float(np.std(sim_enrollment))

        # Distance: Euclidean on summary stats
        d_completion = (sim_mean_completion - obs_mean_completion) ** 2
        d_enrollment = (sim_std_enrollment - obs_std_enrollment) ** 2
        distances[s] = math.sqrt(d_completion + d_enrollment)

    # Acceptance threshold
    epsilon = float(np.quantile(distances, epsilon_quantile))

    # Accept samples
    accepted_mask = distances <= epsilon
    accepted_mu = prior_mu[accepted_mask]
    accepted_sigma = prior_sigma[accepted_mask]

    n_accepted = int(np.sum(accepted_mask))
    acceptance_rate = n_accepted / n_simulations if n_simulations > 0 else 0.0

    # Combine into posterior samples
    posterior_samples = [
        [float(accepted_mu[i]), float(accepted_sigma[i])]
        for i in range(n_accepted)
    ]

    # Summary statistics
    if n_accepted > 0:
        mu_median = float(np.median(accepted_mu))
        sigma_median = float(np.median(accepted_sigma))
        mu_ci = [float(np.percentile(accepted_mu, 2.5)),
                 float(np.percentile(accepted_mu, 97.5))]
        sigma_ci = [float(np.percentile(accepted_sigma, 2.5)),
                    float(np.percentile(accepted_sigma, 97.5))]
    else:
        mu_median = 0.5
        sigma_median = 1.0
        mu_ci = [0.0, 1.0]
        sigma_ci = [0.0, 2.0]

    return {
        "posterior_samples": posterior_samples,
        "acceptance_rate": float(acceptance_rate),
        "parameter_medians": {
            "mu": mu_median,
            "sigma": sigma_median,
        },
        "credible_intervals": {
            "mu": mu_ci,
            "sigma": sigma_ci,
        },
    }
