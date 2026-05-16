"""Shared fixtures for EnrollmentOracle tests."""

import json
import os
import sys
import pytest

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

FIXTURES_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'fixtures')


@pytest.fixture
def sample_ml():
    """Load 20-trial ML fixture data."""
    with open(os.path.join(FIXTURES_DIR, 'sample_ml.json')) as f:
        return json.load(f)
