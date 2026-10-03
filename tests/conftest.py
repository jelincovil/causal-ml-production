import pytest

from causal_model.estimator import fit_effect_model
from contracts import load_contract, load_dataset, load_feature_schema, load_thresholds
from monitoring.reference_profile import load_reference_profile


@pytest.fixture(scope="session")
def contract():
    return load_contract()


@pytest.fixture(scope="session")
def thresholds():
    return load_thresholds()


@pytest.fixture(scope="session")
def feature_schema():
    return load_feature_schema()


@pytest.fixture(scope="session")
def df():
    return load_dataset()


@pytest.fixture(scope="session")
def reference():
    return load_reference_profile()


@pytest.fixture(scope="session")
def fitted(df, contract):
    return fit_effect_model(df, contract)
