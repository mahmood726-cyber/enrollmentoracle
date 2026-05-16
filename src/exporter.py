"""Export trained model as JSON for browser-side inference."""

import json


def export_model_json(model, metrics):
    """Export model parameters as JSON dict.

    Exports feature importances, metrics, and model parameters for
    potential browser-side inference or analysis.

    Args:
        model: Trained classifier.
        metrics: Metrics dict from train_model().

    Returns:
        JSON-serializable dict.
    """
    data = {
        "feature_names": metrics.get("feature_names", []),
        "metrics": {k: v for k, v in metrics.items() if k != "feature_names"},
    }

    # Extract feature importances
    importances = None
    for obj in [model, getattr(model, "estimator", None)]:
        if obj is not None and hasattr(obj, "feature_importances_"):
            importances = obj.feature_importances_.tolist()
            break
    data["feature_importances"] = importances or []

    # Export basic model params
    try:
        params = model.get_params()
        data["params"] = {k: str(v) for k, v in params.items()
                         if not callable(v)}
    except Exception:
        data["params"] = {}

    return data
