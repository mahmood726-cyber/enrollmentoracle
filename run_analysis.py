#!/usr/bin/env python
"""EnrollmentOracle pipeline runner.

Fetches completed/terminated/withdrawn interventional trials from CT.gov,
engineers features, trains a completion prediction model, calibrates it,
computes feature importances, and exports dashboard_data.json.
"""

import json
import os
import sys
import time

# Add src to path for absolute imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from feature_engineer import extract_features, FEATURE_NAMES
from label_generator import generate_labels
from model_trainer import train_model, evaluate_model
from calibrator import calibration_metrics
from explainer import compute_feature_importance
from exporter import export_model_json


CACHE_DIR = os.path.join(os.path.dirname(__file__), 'data', 'raw')
CACHE_FILE = os.path.join(CACHE_DIR, 'trials_cache.json')
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'data', 'processed')
OUTPUT_FILE = os.path.join(OUTPUT_DIR, 'dashboard_data.json')
FIXTURE_FILE = os.path.join(os.path.dirname(__file__), 'data', 'fixtures', 'sample_ml.json')

# Cache expiry: 24 hours
CACHE_TTL_SECONDS = 86400
MAX_TRIALS = 200


def load_cached_trials():
    """Load trials from cache if fresh enough.

    Returns:
        List of trial dicts, or None if cache is stale/missing.
    """
    if not os.path.exists(CACHE_FILE):
        return None
    try:
        mtime = os.path.getmtime(CACHE_FILE)
        if time.time() - mtime > CACHE_TTL_SECONDS:
            print("[pipeline] Cache expired, will re-fetch.")
            return None
        with open(CACHE_FILE, 'r', encoding='utf-8') as f:
            trials = json.load(f)
        print(f"[pipeline] Loaded {len(trials)} trials from cache.")
        return trials
    except Exception as e:
        print(f"[pipeline] Cache load error: {e}")
        return None


def save_cache(trials):
    """Save trials to cache file."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    with open(CACHE_FILE, 'w', encoding='utf-8') as f:
        json.dump(trials, f, indent=2)
    print(f"[pipeline] Cached {len(trials)} trials.")


def fetch_trials():
    """Fetch trials from CT.gov API (with cache check).

    Falls back to fixture data if API fetch fails or returns nothing.
    """
    # Check cache first
    cached = load_cached_trials()
    if cached is not None:
        return cached

    # Fetch from CT.gov
    print(f"[pipeline] Fetching up to {MAX_TRIALS} trials from ClinicalTrials.gov...")
    try:
        from harvester import fetch_trials_for_ml
        trials = fetch_trials_for_ml(max_trials=MAX_TRIALS)
        if trials:
            save_cache(trials)
            print(f"[pipeline] Fetched {len(trials)} trials from API.")
            return trials
        else:
            print("[pipeline] API returned no trials, falling back to fixtures.")
    except Exception as e:
        print(f"[pipeline] API fetch failed: {e}")
        print("[pipeline] Falling back to fixture data.")

    # Fallback: load fixtures
    if os.path.exists(FIXTURE_FILE):
        with open(FIXTURE_FILE, 'r', encoding='utf-8') as f:
            trials = json.load(f)
        print(f"[pipeline] Loaded {len(trials)} trials from fixtures.")
        return trials

    print("[pipeline] ERROR: No data available (no cache, API failed, no fixtures).")
    sys.exit(1)


def build_sponsor_report(trials, eval_results):
    """Build sponsor completion report card.

    Args:
        trials: List of trial dicts.
        eval_results: List of evaluation result dicts from evaluate_model.

    Returns:
        List of sponsor report dicts sorted by delta (actual - expected).
    """
    # Map nctId -> eval result
    eval_map = {r['nctId']: r for r in eval_results}

    # Aggregate by sponsor
    sponsors = {}
    for trial in trials:
        sponsor = trial.get('sponsorClass', 'OTHER') or 'OTHER'
        nct_id = trial.get('nctId', '')
        ev = eval_map.get(nct_id)
        if ev is None:
            continue

        if sponsor not in sponsors:
            sponsors[sponsor] = {
                'sponsor': sponsor,
                'n_trials': 0,
                'actual_completed': 0,
                'sum_predicted': 0.0,
            }
        sponsors[sponsor]['n_trials'] += 1
        sponsors[sponsor]['actual_completed'] += ev['actual']
        sponsors[sponsor]['sum_predicted'] += ev['prob_complete']

    report = []
    for s in sponsors.values():
        n = s['n_trials']
        actual_rate = s['actual_completed'] / n if n > 0 else 0
        expected_rate = s['sum_predicted'] / n if n > 0 else 0
        delta = actual_rate - expected_rate
        report.append({
            'sponsor': s['sponsor'],
            'n_trials': n,
            'actual_rate': round(actual_rate, 4),
            'expected_rate': round(expected_rate, 4),
            'delta': round(delta, 4),
        })

    report.sort(key=lambda x: x['delta'])
    return report


def build_calibration_data(eval_results, n_bins=10):
    """Build calibration curve data (predicted vs actual by decile bin).

    Args:
        eval_results: List of dicts with prob_complete and actual.
        n_bins: Number of bins for calibration curve.

    Returns:
        List of dicts with bin_center, mean_predicted, mean_actual, count.
    """
    bins = [[] for _ in range(n_bins)]
    for r in eval_results:
        p = r['prob_complete']
        idx = min(int(p * n_bins), n_bins - 1)
        bins[idx].append(r)

    cal_data = []
    for i in range(n_bins):
        if not bins[i]:
            continue
        preds = [r['prob_complete'] for r in bins[i]]
        actuals = [r['actual'] for r in bins[i]]
        cal_data.append({
            'bin_center': round((i + 0.5) / n_bins, 3),
            'mean_predicted': round(sum(preds) / len(preds), 4),
            'mean_actual': round(sum(actuals) / len(actuals), 4),
            'count': len(bins[i]),
        })

    return cal_data


def build_roc_data(eval_results, n_thresholds=50):
    """Build approximate ROC curve data.

    Args:
        eval_results: List of dicts with prob_complete and actual.
        n_thresholds: Number of threshold points.

    Returns:
        List of dicts with threshold, fpr, tpr.
    """
    roc_points = []
    for i in range(n_thresholds + 1):
        threshold = i / n_thresholds
        tp = fp = tn = fn = 0
        for r in eval_results:
            pred = 1 if r['prob_complete'] >= threshold else 0
            actual = r['actual']
            if pred == 1 and actual == 1:
                tp += 1
            elif pred == 1 and actual == 0:
                fp += 1
            elif pred == 0 and actual == 0:
                tn += 1
            else:
                fn += 1
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        roc_points.append({
            'threshold': round(threshold, 3),
            'fpr': round(fpr, 4),
            'tpr': round(tpr, 4),
        })

    return roc_points


def build_population_data(trials, eval_results):
    """Build population histogram data.

    Args:
        trials: List of trial dicts.
        eval_results: List of evaluation result dicts.

    Returns:
        List of trial summaries with probability, condition, phase.
    """
    eval_map = {r['nctId']: r for r in eval_results}
    population = []
    for trial in trials:
        nct_id = trial.get('nctId', '')
        ev = eval_map.get(nct_id)
        if ev is None:
            continue
        population.append({
            'nctId': nct_id,
            'prob_complete': round(ev['prob_complete'], 4),
            'condition': trial.get('condition', 'Unknown'),
            'phase': trial.get('phase', 'NA'),
            'actual': ev['actual'],
        })
    return population


def run_pipeline():
    """Execute the full EnrollmentOracle pipeline."""
    print("=" * 60)
    print("EnrollmentOracle Pipeline")
    print("=" * 60)

    # Step 1: Fetch data
    print("\n[1/6] Fetching trial data...")
    trials = fetch_trials()
    print(f"  -> {len(trials)} trials loaded.")

    # Step 2: Feature engineering (validate features extract correctly)
    print("\n[2/6] Feature engineering...")
    for trial in trials[:3]:
        feats = extract_features(trial)
        assert len(feats) == len(FEATURE_NAMES), \
            f"Feature count mismatch: {len(feats)} vs {len(FEATURE_NAMES)}"
    labels = generate_labels(trials)
    n_completed = sum(1 for lb in labels if lb['completed'] == 1)
    n_failed = len(labels) - n_completed
    print(f"  -> {n_completed} completed, {n_failed} terminated/withdrawn.")

    # Step 3: Train model
    print("\n[3/6] Training model...")
    model, metrics = train_model(trials, n_folds=3, seed=42)
    print(f"  -> Classifier: {metrics['classifier_type']}")
    print(f"  -> CV Accuracy: {metrics['accuracy']:.4f}")
    if metrics['auc'] is not None:
        print(f"  -> CV AUC: {metrics['auc']:.4f}")

    # Step 4: Calibration
    print("\n[4/6] Computing calibration metrics...")
    cal_metrics = calibration_metrics(model, trials)
    print(f"  -> Brier score: {cal_metrics['brier_score']:.4f}")
    print(f"  -> Mean predicted: {cal_metrics['mean_predicted']:.4f}")
    print(f"  -> Mean actual: {cal_metrics['mean_actual']:.4f}")

    # Step 5: Feature importance
    print("\n[5/6] Computing feature importances...")
    importances = compute_feature_importance(model, FEATURE_NAMES)
    top5 = importances[:5]
    for fi in top5:
        print(f"  -> {fi['feature']}: {fi['importance']:.4f}")

    # Step 6: Export
    print("\n[6/6] Exporting dashboard data...")
    eval_results = evaluate_model(model, trials)
    model_data = export_model_json(model, metrics)

    # Build all dashboard components
    sponsor_report = build_sponsor_report(trials, eval_results)
    calibration_data = build_calibration_data(eval_results)
    roc_data = build_roc_data(eval_results)
    population_data = build_population_data(trials, eval_results)

    dashboard_data = {
        'model': model_data,
        'metrics': metrics,
        'calibration': cal_metrics,
        'calibration_curve': calibration_data,
        'roc_curve': roc_data,
        'feature_importances': importances,
        'sponsor_report': sponsor_report,
        'population': population_data,
        'trials': [
            {
                'nctId': t.get('nctId', ''),
                'condition': t.get('condition', ''),
                'phase': t.get('phase', ''),
                'sponsorClass': t.get('sponsorClass', ''),
                'enrollmentPlanned': t.get('enrollmentPlanned', 0),
            }
            for t in trials
        ],
        'generated_at': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'n_trials': len(trials),
    }

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(dashboard_data, f, indent=2)
    print(f"  -> Written to {OUTPUT_FILE}")

    print("\n" + "=" * 60)
    print("Pipeline complete.")
    print(f"Open dashboard.html to view results.")
    print("=" * 60)


if __name__ == '__main__':
    run_pipeline()
