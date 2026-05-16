"""Feature importance extraction from tree models."""


def compute_feature_importance(model, feature_names=None):
    """Extract and rank feature importances.

    Args:
        model: Trained tree model with feature_importances_ attribute.
        feature_names: List of feature names. If None, uses indices.

    Returns:
        List of dicts sorted by importance descending:
        [{"feature": name, "importance": float}, ...]
    """
    if feature_names is None:
        from feature_engineer import FEATURE_NAMES
        feature_names = FEATURE_NAMES

    # Try to get importances from model or its base estimator
    importances = None
    for attr_path in [model, getattr(model, "estimator", None),
                      getattr(model, "estimators_", [None])[0] if hasattr(model, "estimators_") else None]:
        if attr_path is not None and hasattr(attr_path, "feature_importances_"):
            importances = attr_path.feature_importances_
            break

    if importances is None:
        return [{"feature": f, "importance": 0.0} for f in feature_names]

    ranked = sorted(
        zip(feature_names, importances),
        key=lambda x: -x[1],
    )
    return [{"feature": f, "importance": float(i)} for f, i in ranked]
