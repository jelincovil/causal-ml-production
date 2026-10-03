import numpy as np

from validation.results import BLOCK, PASS
from validation.schema_checks import check_schema, check_treatment


def test_reference_data_passes(df, contract, feature_schema):
    assert check_schema(df, contract, feature_schema).status == PASS
    assert check_treatment(df, contract).status == PASS


def test_missing_required_column_blocks(df, contract, feature_schema):
    assert check_schema(df.drop(columns=["x3"]), contract, feature_schema).status == BLOCK


def test_missing_outcome_blocks(df, contract, feature_schema):
    assert check_schema(df.drop(columns=["outcome"]), contract, feature_schema).status == BLOCK


def test_unknown_column_blocks(df, contract, feature_schema):
    assert check_schema(df.assign(extra=1), contract, feature_schema).status == BLOCK


def test_non_numeric_column_blocks(df, contract, feature_schema):
    assert check_schema(df.assign(x1="a"), contract, feature_schema).status == BLOCK


def test_nulls_in_outcome_block(df, contract, feature_schema):
    d = df.copy()
    d.loc[0, "outcome"] = np.nan
    assert check_schema(d, contract, feature_schema).status == BLOCK


def test_nulls_in_covariate_do_not_block_schema(df, contract, feature_schema):
    d = df.copy()
    d.loc[0, "x1"] = np.nan
    assert check_schema(d, contract, feature_schema).status == PASS


def test_invalid_treatment_value_blocks(df, contract):
    assert check_treatment(df.assign(treatment=2), contract).status == BLOCK


def test_single_treatment_level_blocks(df, contract):
    assert check_treatment(df.assign(treatment=1), contract).status == BLOCK
