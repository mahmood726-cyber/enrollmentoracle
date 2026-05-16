"""Generate binary and continuous labels from trial outcomes."""

from datetime import datetime


def generate_labels(trials):
    """Generate labels for a list of trials.

    Args:
        trials: List of trial dicts with status, enrollmentPlanned,
                enrollmentActual, startDate, primaryCompletionDate.

    Returns:
        List of label dicts, each containing:
            - nctId: trial identifier
            - completed: 1 if COMPLETED, 0 if TERMINATED/WITHDRAWN
            - enrollment_ratio: enrollmentActual / enrollmentPlanned (0 if planned is 0)
            - months_to_completion: months from startDate to primaryCompletionDate
              (None if either date is missing or unparseable)
    """
    labels = []
    for trial in trials:
        status = trial.get("status", "").upper()
        nct_id = trial.get("nctId", "")

        # Binary label
        completed = 1 if status == "COMPLETED" else 0

        # Enrollment ratio
        planned = trial.get("enrollmentPlanned", 0) or 0
        actual = trial.get("enrollmentActual", 0) or 0
        if planned > 0:
            enrollment_ratio = actual / planned
        else:
            enrollment_ratio = 0.0

        # Months to completion
        months = _compute_months(
            trial.get("startDate"),
            trial.get("primaryCompletionDate"),
        )

        labels.append({
            "nctId": nct_id,
            "completed": completed,
            "enrollment_ratio": round(enrollment_ratio, 4),
            "months_to_completion": round(months, 1) if months is not None else None,
        })

    return labels


def _compute_months(start_str, end_str):
    """Compute months between two date strings.

    Supports formats: YYYY-MM-DD, YYYY-MM, YYYY.

    Returns:
        Float months, or None if either date is missing/unparseable.
    """
    if not start_str or not end_str:
        return None

    start = _parse_date(start_str)
    end = _parse_date(end_str)

    if start is None or end is None:
        return None

    delta = end - start
    months = delta.days / 30.44  # average days per month
    return max(months, 0.0)


def _parse_date(date_str):
    """Parse a date string in various formats.

    Returns:
        datetime object, or None if unparseable.
    """
    if not date_str:
        return None

    for fmt in ("%Y-%m-%d", "%Y-%m", "%Y"):
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue
    return None
