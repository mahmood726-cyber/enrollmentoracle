"""Fetch completed/terminated/withdrawn trials from ClinicalTrials.gov API v2."""

import json
import urllib.request
import urllib.parse
from pathlib import Path


BASE_URL = "https://clinicaltrials.gov/api/v2/studies"


def fetch_trials_for_ml(max_trials=100, statuses=None):
    """Fetch interventional trials with known outcomes from CT.gov API v2.

    Args:
        max_trials: Maximum number of trials to fetch.
        statuses: List of statuses to include. Defaults to COMPLETED, TERMINATED, WITHDRAWN.

    Returns:
        List of parsed trial dicts ready for feature engineering.
    """
    if statuses is None:
        statuses = ["COMPLETED", "TERMINATED", "WITHDRAWN"]

    trials = []
    page_token = None

    while len(trials) < max_trials:
        params = {
            "format": "json",
            "query.term": "AREA[StudyType]INTERVENTIONAL",
            "filter.overallStatus": "|".join(statuses),
            "pageSize": min(50, max_trials - len(trials)),
            "fields": (
                "NCTId,OverallStatus,Phase,LeadSponsorClass,"
                "Condition,EnrollmentCount,EnrollmentType,"
                "NumberOfArms,DesignMasking,MinimumAge,MaximumAge,"
                "Sex,HealthyVolunteers,PrimaryOutcomeCount,"
                "SecondaryOutcomeCount,EligibilityCriteria,"
                "StudyFirstPostDate,PrimaryCompletionDate,StartDate,"
                "LocationCountry,LocationFacility"
            ),
        }
        if page_token:
            params["pageToken"] = page_token

        url = f"{BASE_URL}?{urllib.parse.urlencode(params)}"

        try:
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except Exception as e:
            print(f"[harvester] API error: {e}")
            break

        studies = data.get("studies", [])
        if not studies:
            break

        for study in studies:
            parsed = parse_api_trial_ml(study)
            if parsed is not None:
                trials.append(parsed)

        page_token = data.get("nextPageToken")
        if not page_token:
            break

    return trials[:max_trials]


def parse_api_trial_ml(raw):
    """Extract fields needed for ML feature engineering from a CT.gov API v2 study.

    Args:
        raw: Raw study dict from the API.

    Returns:
        Dict with standardized fields, or None if essential fields are missing.
    """
    try:
        proto = raw.get("protocolSection", {})
        ident = proto.get("identificationModule", {})
        status_mod = proto.get("statusModule", {})
        design = proto.get("designModule", {})
        eligibility = proto.get("eligibilityModule", {})
        sponsor_mod = proto.get("sponsorCollaboratorsModule", {})
        outcomes_mod = proto.get("outcomesModule", {})
        contacts_locs = proto.get("contactsLocationsModule", {})

        nct_id = ident.get("nctId")
        overall_status = status_mod.get("overallStatus")

        if not nct_id or not overall_status:
            return None

        # Enrollment
        enrollment_info = design.get("enrollmentInfo", {})
        enrollment_count = enrollment_info.get("count", 0)

        # Phase
        phases = design.get("phases", [])
        phase_str = phases[0] if phases else ""

        # Sponsor
        lead_sponsor = sponsor_mod.get("leadSponsor", {})
        sponsor_class = lead_sponsor.get("class", "OTHER")

        # Conditions
        conditions_mod = proto.get("conditionsModule", {})
        conditions = conditions_mod.get("conditions", [])

        # Design
        design_info = design.get("designInfo", {})
        n_arms = design.get("numberOfArms", 1)
        masking_info = design_info.get("maskingInfo", {})
        masking = masking_info.get("masking", "NONE")

        # Eligibility
        min_age = eligibility.get("minimumAge", "")
        max_age = eligibility.get("maximumAge", "")
        sex = eligibility.get("sex", "ALL")
        healthy = eligibility.get("healthyVolunteers", False)
        criteria_text = eligibility.get("eligibilityCriteria", "")

        # Outcomes counts
        primary_outcomes = outcomes_mod.get("primaryOutcomes", [])
        secondary_outcomes = outcomes_mod.get("secondaryOutcomes", [])

        # Locations
        locations = contacts_locs.get("locations", [])
        countries = set()
        for loc in locations:
            country = loc.get("country")
            if country:
                countries.add(country)

        # Dates
        start_date_struct = status_mod.get("startDateStruct", {})
        start_date = start_date_struct.get("date", "")
        pc_date_struct = status_mod.get("primaryCompletionDateStruct", {})
        pc_date = pc_date_struct.get("date", "")
        first_post = status_mod.get("studyFirstPostDateStruct", {}).get("date", "")

        return {
            "nctId": nct_id,
            "status": overall_status,
            "phase": phase_str,
            "sponsorClass": sponsor_class,
            "condition": conditions[0] if conditions else "",
            "enrollmentPlanned": enrollment_count,
            "enrollmentActual": enrollment_count,  # API doesn't always split planned/actual
            "sitesCount": len(locations) if locations else 1,
            "countriesCount": len(countries) if countries else 1,
            "arms": n_arms or 1,
            "masking": masking,
            "minAge": min_age,
            "maxAge": max_age,
            "sex": sex,
            "healthyVolunteers": healthy,
            "primaryOutcomesCount": len(primary_outcomes),
            "secondaryOutcomesCount": len(secondary_outcomes),
            "eligibilityLength": len(criteria_text),
            "studyFirstPostDate": first_post,
            "primaryCompletionDate": pc_date if pc_date else None,
            "startDate": start_date,
        }

    except Exception as e:
        print(f"[harvester] Parse error: {e}")
        return None


def load_fixture_trials(fixture_path=None):
    """Load trials from a local JSON fixture file.

    Args:
        fixture_path: Path to JSON file. Defaults to data/fixtures/sample_ml.json.

    Returns:
        List of trial dicts.
    """
    if fixture_path is None:
        fixture_path = Path(__file__).parent.parent / "data" / "fixtures" / "sample_ml.json"
    else:
        fixture_path = Path(fixture_path)

    with open(fixture_path, "r", encoding="utf-8") as f:
        return json.load(f)
