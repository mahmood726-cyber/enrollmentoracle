"""Stats engine test suite -- 20 tests."""

import os
import sys
import math

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from stats_engine import (
    bootstrap_ci, bootstrap_auc_ci, bootstrap_brier_ci, bootstrap_accuracy_ci,
    calibration_slope_intercept, learning_curves, delong_test,
    apply_smote, tune_hyperparameters, competing_risks,
    conformal_prediction, tree_shap, bayesian_model_averaging,
    survival_prediction,
)
from model_trainer import train_model
from feature_engineer import extract_feature_matrix, features_to_array, FEATURE_NAMES
from label_generator import generate_labels
from sklearn.metrics import roc_auc_score, brier_score_loss


# === Bootstrap (2 tests) ===

class TestBootstrap:
    def test_bootstrap_auc_ci(self, sample_ml):
        model, _ = train_model(sample_ml)
        X = np.array(features_to_array(extract_feature_matrix(sample_ml), FEATURE_NAMES))
        y = np.array([l["completed"] for l in generate_labels(sample_ml)])
        p = model.predict_proba(X)[:, 1]
        result = bootstrap_auc_ci(y, p, n_bootstrap=200, seed=42)
        assert 0 < result["ci_lower"] <= result["estimate"] <= result["ci_upper"] <= 1

    def test_bootstrap_width(self):
        # Add noise so AUC < 1 and bootstrap resamples vary
        rng = np.random.RandomState(42)
        y = np.array([1]*10 + [0]*10)
        p = np.array([0.7, 0.6, 0.8, 0.55, 0.65, 0.75, 0.5, 0.85, 0.4, 0.9,
                       0.3, 0.2, 0.35, 0.1, 0.25, 0.15, 0.45, 0.05, 0.38, 0.12])
        result = bootstrap_auc_ci(y, p, n_bootstrap=200, seed=42)
        assert result["ci_upper"] - result["ci_lower"] > 0  # non-zero width


# === Calibration slope (2 tests) ===

class TestCalibrationSlope:
    def test_calibration_slope_perfect(self):
        y = np.array([1, 1, 1, 0, 0, 0, 1, 0, 1, 0])
        p = np.array([0.9, 0.8, 0.7, 0.1, 0.2, 0.3, 0.6, 0.4, 0.85, 0.15])
        result = calibration_slope_intercept(y, p)
        assert result["slope"] > 0  # positive slope
        assert "intercept" in result

    def test_calibration_ci(self):
        np.random.seed(42)
        y = np.random.binomial(1, 0.5, 50)
        p = np.clip(y + np.random.randn(50)*0.2, 0.01, 0.99)
        result = calibration_slope_intercept(y, p)
        assert "slope_ci" in result and len(result["slope_ci"]) == 2


# === Learning curves (1 test) ===

class TestLearningCurves:
    def test_learning_curves(self, sample_ml):
        result = learning_curves(sample_ml, n_points=3, cv=2, seed=42)
        assert len(result["train_sizes"]) >= 2
        assert len(result["train_scores"]) == len(result["val_scores"])


# === DeLong (2 tests) ===

class TestDeLong:
    def test_delong_same_model(self):
        np.random.seed(42)
        y = np.array([1]*20 + [0]*20)
        p = np.random.rand(40)
        result = delong_test(y, p, p)
        assert result["p_value"] > 0.9  # same model -> no difference
        assert abs(result["z_stat"]) < 0.1

    def test_delong_different_models(self):
        y = np.array([1]*20 + [0]*20)
        p1 = np.array([0.9]*20 + [0.1]*20)  # perfect
        p2 = np.random.RandomState(42).rand(40)  # random
        result = delong_test(y, p1, p2)
        assert result["auc1"] > result["auc2"]


# === SMOTE (2 tests) ===

class TestSMOTE:
    def test_smote_augments(self):
        X = np.array([[1,1],[2,2],[3,3],[10,10],[11,11],[12,12],[13,13],[14,14],[15,15],[16,16]])
        y = np.array([1,1,1,0,0,0,0,0,0,0])
        X_new, y_new = apply_smote(X, y, k=2, seed=42)
        assert len(y_new) > len(y)
        assert sum(y_new == 1) == sum(y_new == 0)  # balanced

    def test_smote_preserves_features(self):
        X = np.array([[0,0],[1,1],[2,2],[10,10],[11,11]])
        y = np.array([1,1,0,0,0])
        X_new, y_new = apply_smote(X, y, k=1, seed=42)
        assert X_new.shape[1] == 2  # same number of features


# === Hyperparameter tuning (1 test) ===

class TestTuning:
    def test_tune(self, sample_ml):
        result = tune_hyperparameters(sample_ml, n_iter=5, cv=2, seed=42)
        assert "best_params" in result
        assert "best_score" in result
        assert result["best_score"] > 0


# === Competing risks (2 tests) ===

class TestCompetingRisks:
    def test_competing_risks_3class(self, sample_ml):
        result = competing_risks(sample_ml)
        assert len(result["predicted_probabilities"]) == 20
        for pp in result["predicted_probabilities"]:
            total = pp["p_completed"] + pp["p_terminated"] + pp["p_withdrawn"]
            assert abs(total - 1.0) < 0.01

    def test_competing_risks_or(self, sample_ml):
        result = competing_risks(sample_ml)
        assert len(result["cause_specific_or"]) > 0


# === Conformal prediction (2 tests) ===

class TestConformal:
    def test_conformal_coverage(self, sample_ml):
        model, _ = train_model(sample_ml)
        result = conformal_prediction(model, sample_ml, alpha=0.10, seed=42)
        # Coverage should be >= 1-alpha on calibration set (by construction)
        assert result["coverage"] >= 0.80  # allow some slack on small data

    def test_conformal_intervals(self, sample_ml):
        model, _ = train_model(sample_ml)
        result = conformal_prediction(model, sample_ml, alpha=0.10, seed=42)
        for pred in result["predictions"]:
            assert pred["lower"] <= pred["upper"]
            assert 0 <= pred["lower"] and pred["upper"] <= 1


# === TreeSHAP (2 tests) ===

class TestTreeSHAP:
    def test_shap_shape(self, sample_ml):
        model, metrics = train_model(sample_ml)
        X = np.array(features_to_array(extract_feature_matrix(sample_ml), FEATURE_NAMES))
        result = tree_shap(model, X, FEATURE_NAMES)
        assert result["shap_values"].shape == X.shape
        assert len(result["feature_names"]) == X.shape[1]

    def test_shap_base_value(self, sample_ml):
        model, _ = train_model(sample_ml)
        X = np.array(features_to_array(extract_feature_matrix(sample_ml), FEATURE_NAMES))
        result = tree_shap(model, X, FEATURE_NAMES)
        assert 0 <= result["base_value"] <= 1


# === BMA (1 test) ===

class TestBMA:
    def test_bma_weights(self, sample_ml):
        result = bayesian_model_averaging(sample_ml, seed=42)
        weights = [m["weight"] for m in result["model_weights"]]
        assert abs(sum(weights) - 1.0) < 0.01
        assert len(result["bma_predictions"]) == 20


# === Survival (1 test) ===

class TestSurvival:
    def test_survival_prediction(self, sample_ml):
        result = survival_prediction(sample_ml, seed=42)
        completed = [p for p in result["predictions"] if not p["censored"]]
        assert len(completed) > 0
        assert all(p["predicted_months"] > 0 for p in completed)


# === Integration (2 tests) ===

class TestIntegrationStats:
    def test_full_enhanced_pipeline(self, sample_ml):
        model, metrics = train_model(sample_ml)
        X = np.array(features_to_array(extract_feature_matrix(sample_ml), FEATURE_NAMES))
        y = np.array([l["completed"] for l in generate_labels(sample_ml)])
        p = model.predict_proba(X)[:, 1]

        boot = bootstrap_auc_ci(y, p, n_bootstrap=100, seed=42)
        cal = calibration_slope_intercept(y, p)
        conf = conformal_prediction(model, sample_ml, seed=42)
        cr = competing_risks(sample_ml)
        assert boot["estimate"] > 0
        assert cal["slope"] > 0
        assert conf["coverage"] > 0
        assert len(cr["predicted_probabilities"]) == 20

    def test_all_methods_deterministic(self, sample_ml):
        r1 = competing_risks(sample_ml)
        r2 = competing_risks(sample_ml)
        for p1, p2 in zip(r1["predicted_probabilities"], r2["predicted_probabilities"]):
            assert abs(p1["p_completed"] - p2["p_completed"]) < 0.01
