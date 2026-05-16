"""Extract 24 numeric features from clinical trial registry fields."""

import math


# Ordered list of feature names — must match extract_features output
FEATURE_NAMES = [
    "phase_num",
    "enrollment_planned",
    "sites_count",
    "countries_count",
    "arms_count",
    "masking_level",
    "min_age",
    "max_age",
    "age_range",
    "sex_restricted",
    "healthy_volunteers",
    "primary_outcomes_count",
    "secondary_outcomes_count",
    "total_outcomes",
    "eligibility_text_length",
    "year_started",
    "sponsor_industry",
    "sponsor_nih",
    "sponsor_other",
    "sites_per_country",
    "enrollment_per_site",
    "is_multinational",
    "is_large_trial",
    "endpoint_complexity",
]


_PHASE_MAP = {
    "PHASE1": 1,
    "PHASE2": 2,
    "PHASE3": 3,
    "PHASE4": 4,
    "EARLY_PHASE1": 0.5,
    "NA": 0,
}

_MASKING_MAP = {
    "NONE": 0,
    "SINGLE": 1,
    "DOUBLE": 2,
    "TRIPLE": 3,
    "QUADRUPLE": 4,
}


def _parse_age_years(age_str):
    """Convert age string to years.

    Examples:
        "18 Years" -> 18.0
        "6 Months" -> 0.5
        "2 Days" -> ~0.005
        None or "" -> None
    """
    if not age_str or age_str.strip() == "":
        return None

    age_str = age_str.strip()
    parts = age_str.split()
    if len(parts) < 2:
        try:
            return float(parts[0])
        except (ValueError, IndexError):
            return None

    try:
        value = float(parts[0])
    except ValueError:
        return None

    unit = parts[1].upper().rstrip("S")  # "Years" -> "YEAR", "Months" -> "MONTH"

    if unit == "YEAR":
        return value
    elif unit == "MONTH":
        return value / 12.0
    elif unit == "WEEK":
        return value / 52.0
    elif unit == "DAY":
        return value / 365.25
    elif unit == "HOUR":
        return value / (365.25 * 24)
    else:
        return value  # assume years


def extract_features(trial):
    """Extract 24 numeric features from a trial dict.

    Args:
        trial: Dict with standardized trial fields (from harvester or fixtures).

    Returns:
        Dict mapping feature name -> numeric value. No NaN values; missing
        values are replaced with sensible defaults (0 or median-like values).
    """
    # Phase
    phase_str = trial.get("phase", "")
    phase_num = _PHASE_MAP.get(phase_str, 0)

    # Enrollment
    enrollment_planned = trial.get("enrollmentPlanned", 0) or 0

    # Sites and countries
    sites_count = trial.get("sitesCount", 1) or 1
    countries_count = trial.get("countriesCount", 1) or 1

    # Arms
    arms_count = trial.get("arms", 1) or 1

    # Masking
    masking_str = trial.get("masking", "NONE") or "NONE"
    masking_level = _MASKING_MAP.get(masking_str.upper(), 0)

    # Age
    min_age = _parse_age_years(trial.get("minAge"))
    max_age = _parse_age_years(trial.get("maxAge"))

    min_age_val = min_age if min_age is not None else 18.0  # default adult
    max_age_val = max_age if max_age is not None else 100.0  # no upper limit
    age_range = max_age_val - min_age_val

    # Sex restriction
    sex = trial.get("sex", "ALL") or "ALL"
    sex_restricted = 0 if sex.upper() == "ALL" else 1

    # Healthy volunteers
    hv = trial.get("healthyVolunteers", False)
    healthy_volunteers = 1 if hv else 0

    # Outcomes
    primary_outcomes_count = trial.get("primaryOutcomesCount", 0) or 0
    secondary_outcomes_count = trial.get("secondaryOutcomesCount", 0) or 0
    total_outcomes = primary_outcomes_count + secondary_outcomes_count

    # Eligibility text length
    eligibility_text_length = trial.get("eligibilityLength", 0) or 0

    # Year started
    start_date = trial.get("startDate", "")
    if start_date and len(start_date) >= 4:
        try:
            year_started = int(start_date[:4])
        except ValueError:
            year_started = 2020
    else:
        year_started = 2020

    # Sponsor type (one-hot)
    sponsor_class = (trial.get("sponsorClass", "OTHER") or "OTHER").upper()
    sponsor_industry = 1 if sponsor_class == "INDUSTRY" else 0
    sponsor_nih = 1 if sponsor_class == "NIH" else 0
    sponsor_other = 1 if sponsor_class not in ("INDUSTRY", "NIH") else 0

    # Derived features
    sites_per_country = sites_count / max(countries_count, 1)
    enrollment_per_site = enrollment_planned / max(sites_count, 1)
    is_multinational = 1 if countries_count > 1 else 0
    is_large_trial = 1 if enrollment_planned >= 1000 else 0

    # Endpoint complexity: composite score based on outcomes, arms, masking
    endpoint_complexity = (
        primary_outcomes_count * 2.0
        + secondary_outcomes_count * 0.5
        + arms_count * 0.3
        + masking_level * 0.2
    )

    features = {
        "phase_num": phase_num,
        "enrollment_planned": enrollment_planned,
        "sites_count": sites_count,
        "countries_count": countries_count,
        "arms_count": arms_count,
        "masking_level": masking_level,
        "min_age": min_age_val,
        "max_age": max_age_val,
        "age_range": age_range,
        "sex_restricted": sex_restricted,
        "healthy_volunteers": healthy_volunteers,
        "primary_outcomes_count": primary_outcomes_count,
        "secondary_outcomes_count": secondary_outcomes_count,
        "total_outcomes": total_outcomes,
        "eligibility_text_length": eligibility_text_length,
        "year_started": year_started,
        "sponsor_industry": sponsor_industry,
        "sponsor_nih": sponsor_nih,
        "sponsor_other": sponsor_other,
        "sites_per_country": sites_per_country,
        "enrollment_per_site": enrollment_per_site,
        "is_multinational": is_multinational,
        "is_large_trial": is_large_trial,
        "endpoint_complexity": endpoint_complexity,
    }

    # Safety: ensure no NaN/inf values
    for key in features:
        val = features[key]
        if val is None or (isinstance(val, float) and (math.isnan(val) or math.isinf(val))):
            features[key] = 0.0

    return features


def extract_feature_matrix(trials):
    """Extract features for multiple trials as a list of dicts.

    Args:
        trials: List of trial dicts.

    Returns:
        List of feature dicts, one per trial.
    """
    return [extract_features(t) for t in trials]


def features_to_array(feature_dicts, feature_names=None):
    """Convert list of feature dicts to 2D list (for sklearn).

    Args:
        feature_dicts: List of dicts from extract_features().
        feature_names: Ordered list of feature names. Defaults to FEATURE_NAMES.

    Returns:
        2D list of shape (n_trials, n_features).
    """
    if feature_names is None:
        feature_names = FEATURE_NAMES
    return [[fd[name] for name in feature_names] for fd in feature_dicts]
