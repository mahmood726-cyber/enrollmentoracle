"""EnrollmentOracle test suite — 25 tests."""

import json
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from feature_engineer import extract_features, FEATURE_NAMES
from label_generator import generate_labels
from model_trainer import train_model, evaluate_model
from calibrator import calibrate_model, calibration_metrics
from explainer import compute_feature_importance
from predictor import predict_single
from exporter import export_model_json


# === Feature Engineer (5 tests) ===

class TestFeatureEngineer:
    def test_shape_consistency(self, sample_ml):
        """All 20 trials produce feature vectors of same length."""
        features = [extract_features(t) for t in sample_ml]
        lengths = set(len(f) for f in features)
        assert len(lengths) == 1
        assert lengths.pop() == len(FEATURE_NAMES)

    def test_no_nans(self, sample_ml):
        """No NaN values in extracted features."""
        for t in sample_ml:
            feats = extract_features(t)
            for key, val in feats.items():
                assert not (isinstance(val, float) and math.isnan(val)), \
                    f"NaN in {key} for {t['nctId']}"

    def test_phase_encoding(self, sample_ml):
        """Phase is encoded as numeric."""
        feats = extract_features(sample_ml[0])  # PHASE3
        assert feats["phase_num"] == 3

    def test_age_range(self):
        """Age range computed correctly."""
        trial = {
            "minAge": "18 Years", "maxAge": "85 Years", "phase": "PHASE3",
            "sponsorClass": "INDUSTRY", "enrollmentPlanned": 500,
            "sitesCount": 10, "countriesCount": 1, "arms": 2,
            "masking": "DOUBLE", "sex": "ALL", "healthyVolunteers": False,
            "primaryOutcomesCount": 1, "secondaryOutcomesCount": 3,
            "eligibilityLength": 1000, "startDate": "2020-01-01",
        }
        feats = extract_features(trial)
        assert feats["age_range"] == 67  # 85 - 18

    def test_sponsor_encoding(self):
        """Sponsor class one-hot encoded correctly."""
        trial = {
            "phase": "PHASE2", "sponsorClass": "NIH",
            "enrollmentPlanned": 100, "sitesCount": 5, "countriesCount": 1,
            "arms": 1, "masking": "NONE", "minAge": "18 Years",
            "maxAge": "65 Years", "sex": "ALL", "healthyVolunteers": False,
            "primaryOutcomesCount": 1, "secondaryOutcomesCount": 2,
            "eligibilityLength": 800, "startDate": "2020-01-01",
        }
        feats = extract_features(trial)
        assert feats["sponsor_industry"] == 0
        assert feats["sponsor_nih"] == 1
        assert feats["sponsor_other"] == 0


# === Label Generator (4 tests) ===

class TestLabelGenerator:
    def test_all_labeled(self, sample_ml):
        """All 20 trials get labels."""
        labels = generate_labels(sample_ml)
        assert len(labels) == 20

    def test_completed_correct(self, sample_ml):
        """COMPLETED → 1, others → 0."""
        labels = generate_labels(sample_ml)
        for trial, label in zip(sample_ml, labels):
            if trial["status"] == "COMPLETED":
                assert label["completed"] == 1
            else:
                assert label["completed"] == 0

    def test_enrollment_ratio(self, sample_ml):
        """Enrollment ratio = actual / planned."""
        labels = generate_labels(sample_ml)
        # ML001: 4744/4744 = 1.0
        assert abs(labels[0]["enrollment_ratio"] - 1.0) < 0.01

    def test_months_to_completion(self, sample_ml):
        """Months computed from start to completion date."""
        labels = generate_labels(sample_ml)
        # ML001: 2017-04-10 to 2019-03-20 ≈ 23 months
        assert labels[0]["months_to_completion"] is not None
        assert 20 < labels[0]["months_to_completion"] < 30


# === Model Trainer (2 tests) ===

class TestModelTrainer:
    def test_trains_without_error(self, sample_ml):
        """Model trains on fixture data."""
        model, metrics = train_model(sample_ml)
        assert model is not None
        assert "accuracy" in metrics
        assert metrics["n_samples"] == 20

    def test_produces_predictions(self, sample_ml):
        """Trained model produces predictions."""
        model, _ = train_model(sample_ml)
        preds = evaluate_model(model, sample_ml)
        assert len(preds) == 20
        for p in preds:
            assert 0 <= p["prob_complete"] <= 1


# === Calibrator (2 tests) ===

class TestCalibrator:
    def test_calibrate(self, sample_ml):
        """Calibration produces a model."""
        model, _ = train_model(sample_ml)
        cal = calibrate_model(model, sample_ml)
        assert cal is not None

    def test_brier_score(self, sample_ml):
        """Brier score is computed and in [0, 1]."""
        model, _ = train_model(sample_ml)
        metrics = calibration_metrics(model, sample_ml)
        assert "brier_score" in metrics
        assert 0 <= metrics["brier_score"] <= 1


# === Explainer (1 test) ===

class TestExplainer:
    def test_feature_importance(self, sample_ml):
        """Feature importance extracted and ranked."""
        model, metrics = train_model(sample_ml)
        importance = compute_feature_importance(model, metrics["feature_names"])
        assert len(importance) == len(FEATURE_NAMES)
        assert all("feature" in i and "importance" in i for i in importance)
        # Should be sorted descending
        imps = [i["importance"] for i in importance]
        assert imps == sorted(imps, reverse=True)


# === Predictor (2 tests) ===

class TestPredictor:
    def test_predict_single(self, sample_ml):
        """Predict on single trial returns valid result."""
        model, metrics = train_model(sample_ml)
        result = predict_single(model, sample_ml[0], metrics["feature_names"])
        assert "probability" in result
        assert 0 <= result["probability"] <= 1
        assert result["risk"] in ("LOW", "MEDIUM", "HIGH")
        assert "features" in result

    def test_predict_terminated(self, sample_ml):
        """Predict on terminated trial works."""
        model, metrics = train_model(sample_ml)
        # Find a terminated trial
        term = next(t for t in sample_ml if t["status"] == "TERMINATED")
        result = predict_single(model, term, metrics["feature_names"])
        assert 0 <= result["probability"] <= 1


# === Exporter (2 tests) ===

class TestExporter:
    def test_json_export(self, sample_ml):
        """Model exports as JSON-serializable dict."""
        model, metrics = train_model(sample_ml)
        data = export_model_json(model, metrics)
        assert "feature_names" in data
        assert "feature_importances" in data
        json_str = json.dumps(data)
        assert len(json_str) > 50

    def test_deterministic(self, sample_ml):
        """Export is deterministic."""
        model, metrics = train_model(sample_ml)
        j1 = export_model_json(model, metrics)
        j2 = export_model_json(model, metrics)
        assert json.dumps(j1, sort_keys=True) == json.dumps(j2, sort_keys=True)


# === Integration (7 tests) ===

class TestIntegration:
    def test_full_pipeline(self, sample_ml):
        """End-to-end pipeline on fixture data."""
        model, metrics = train_model(sample_ml)
        cal_metrics = calibration_metrics(model, sample_ml)
        importance = compute_feature_importance(model, metrics["feature_names"])
        pred = predict_single(model, sample_ml[0], metrics["feature_names"])

        assert cal_metrics["brier_score"] < 0.5  # better than random
        assert len(importance) >= 15
        assert pred["probability"] > 0

    def test_dashboard_json(self, sample_ml):
        """Dashboard data is JSON-serializable."""
        model, metrics = train_model(sample_ml)
        predictions = evaluate_model(model, sample_ml)
        cal = calibration_metrics(model, sample_ml)
        importance = compute_feature_importance(model, metrics["feature_names"])

        dashboard = {
            "metrics": metrics,
            "calibration": cal,
            "importance": importance,
            "predictions": predictions,
        }
        s = json.dumps(dashboard, default=str)
        assert len(s) > 200
        parsed = json.loads(s)
        assert len(parsed["predictions"]) == 20

    def test_feature_consistency(self, sample_ml):
        """Same input → same features (deterministic)."""
        f1 = extract_features(sample_ml[0])
        f2 = extract_features(sample_ml[0])
        assert f1 == f2

    def test_feature_names_match(self, sample_ml):
        """Model feature names match extract_features output."""
        model, metrics = train_model(sample_ml)
        feats = extract_features(sample_ml[0])
        assert set(metrics["feature_names"]) == set(feats.keys())

    def test_all_statuses_labeled(self, sample_ml):
        """10 COMPLETED, 10 failed."""
        labels = generate_labels(sample_ml)
        completed = sum(1 for l in labels if l["completed"] == 1)
        assert completed == 10
        assert len(labels) - completed == 10

    def test_enrollment_ratio_range(self, sample_ml):
        """Enrollment ratios in reasonable range."""
        labels = generate_labels(sample_ml)
        for l in labels:
            assert 0 <= l["enrollment_ratio"] <= 2.0

    def test_predict_withdrawn_valid(self, sample_ml):
        """Predict on withdrawn trial produces valid output."""
        model, metrics = train_model(sample_ml)
        withdrawn = next(t for t in sample_ml if t["status"] == "WITHDRAWN")
        result = predict_single(model, withdrawn, metrics["feature_names"])
        assert 0 <= result["probability"] <= 1
