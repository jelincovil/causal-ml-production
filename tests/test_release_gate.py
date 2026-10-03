import numpy as np
import pytest

from causal_model.estimator import MODEL_DIR, MODEL_FILE
from validation.release_gate import evaluate_release, validate_runtime_data
from validation.results import BLOCK, DOMAIN, DRIFT, MISSING, PASS, POSITIVITY, SKIPPED, SUTVA, VERSIONS, WARN, CheckResult, REVIEW
from validation.sensitivity_checks import check_sensitivity, robustness_value
from validation.sutva_checks import check_sutva


@pytest.fixture(scope="module")
def metadata():
    import json

    return json.loads((MODEL_DIR / "model_metadata.json").read_text(encoding="utf-8"))


def run(data, contract, reference, metadata, **kw):
    return validate_runtime_data(data, contract, reference, metadata, model_path=MODEL_DIR / MODEL_FILE, **kw)


def test_reference_population_passes(df, contract, reference, metadata):
    v = run(df, contract, reference, metadata)
    assert v.status == PASS and v.effect_estimation == "ALLOWED"
    assert v.get(SUTVA).status == REVIEW


def test_moderate_drift_warns_but_allows(df, contract, reference, metadata):
    v = run(df.assign(x1=df["x1"] + 0.4 * df["x1"].std()), contract, reference, metadata)
    assert v.status == WARN and v.get(DRIFT).status == WARN and not v.has_blocking_error


def test_critical_drift_only_warns_by_contract(df, contract, reference, metadata):
    v = run(df.assign(x2=df["x2"] + 1.0), contract, reference, metadata)
    assert v.get(DRIFT).metrics["level"] == "critical" and v.get(DRIFT).status == WARN


def test_out_of_domain_blocks(df, contract, reference, metadata):
    v = run(df.assign(x0=df["x0"] * 3), contract, reference, metadata)
    assert v.get(DOMAIN).status == BLOCK and v.effect_estimation == "BLOCKED"


def test_nulls_warn_then_block(df, contract, reference, metadata):
    d = df.copy()
    d.loc[d.index[:120], "x5"] = np.nan  # 10%
    assert run(d, contract, reference, metadata).get(MISSING).status == WARN
    d.loc[d.index[:300], "x5"] = np.nan  # 25%
    assert run(d, contract, reference, metadata).get(MISSING).status == BLOCK


def test_broken_overlap_blocks(df, contract, reference, metadata):
    d = df.assign(treatment=(df["x0"] > 0).astype(int))
    assert run(d, contract, reference, metadata).get(POSITIVITY).status == BLOCK


def test_invalid_schema_skips_data_dependent_checks(df, contract, reference, metadata):
    v = run(df.assign(row_id=1), contract, reference, metadata)
    assert v.has_blocking_error
    assert all(v.get(n).status == SKIPPED for n in (MISSING, DOMAIN, POSITIVITY, DRIFT, SUTVA))
    assert v.extra == {}


def test_version_mismatch_blocks(df, contract, reference, metadata):
    bad = {**metadata, "causal_graph_version": "9.9.9"}
    assert run(df, contract, reference, bad).get(VERSIONS).status == BLOCK


def test_tampered_model_hash_blocks(df, contract, reference, metadata):
    bad = {**metadata, "model_sha256": "0" * 64}
    assert run(df, contract, reference, bad).get(VERSIONS).status == BLOCK


def test_sutva_is_never_reported_as_pass(df, contract):
    assert check_sutva(df, contract).status == REVIEW
    clustered = df.assign(g=np.arange(len(df)) % 20)
    c = {**contract, "validation": {**contract["validation"], "sutva": {"cluster_column": "g"}}}
    r = check_sutva(clustered, c)
    assert r.status == REVIEW and "icc_treatment" in r.metrics


def test_robustness_value_properties():
    assert robustness_value(0.0, 1000) == 0.0
    assert robustness_value(10, 1000) > robustness_value(5, 1000) > 0
    assert robustness_value(5, 1000, alpha=0.05) < robustness_value(5, 1000, alpha=1.0)
    assert robustness_value(10.8, 1173, alpha=1.0) == pytest.approx(0.27, abs=0.02)


def test_sensitivity_status_follows_threshold(thresholds):
    assert check_sensitivity({"t_stat": 12.0, "dof": 1000}, thresholds).status == "ROBUST"
    assert check_sensitivity({"t_stat": 2.5, "dof": 1000}, thresholds).status == REVIEW


def _release_kwargs(contract, thresholds, df, reference, metadata):
    refs = {k: {"status": "PASS", "original_effect": 1, "refuted_effect": 0, "p_value": 0.5}
            for k in ("placebo_treatment", "random_common_cause", "data_subset")}
    return dict(
        contract=contract, thresholds=thresholds,
        identification={"identified": True},
        reference_validation=run(df, contract, reference, metadata),
        refutation={"refutations": refs},
        sensitivity=CheckResult("s", "ROBUST", "", {"rv_q1_alpha05": 0.2}),
        synthetic={"passed": True, "abs_error": 0.0, "max_abs_error": 0.4},
        consistency_problems=[], model_card_errors=[],
    )


def test_release_ok_when_all_conditions_hold(contract, thresholds, df, reference, metadata):
    ok, failures = evaluate_release(**_release_kwargs(contract, thresholds, df, reference, metadata))
    assert ok and failures == []


def test_release_blocked_by_failed_required_refutation(contract, thresholds, df, reference, metadata):
    kw = _release_kwargs(contract, thresholds, df, reference, metadata)
    kw["refutation"]["refutations"]["placebo_treatment"]["status"] = "FAIL"
    ok, failures = evaluate_release(**kw)
    assert not ok and any("placebo_treatment" in f for f in failures)


@pytest.mark.parametrize(
    "key,value,fragment",
    [
        ("identification", {"identified": False}, "identificable"),
        ("synthetic", {"passed": False, "abs_error": 1.0, "max_abs_error": 0.4}, "sintetica"),
        ("consistency_problems", ["hash distinto"], "hash"),
        ("model_card_errors", ["falta campo"], "model card"),
    ],
)
def test_release_blocked_by_each_mandatory_condition(contract, thresholds, df, reference, metadata, key, value, fragment):
    kw = _release_kwargs(contract, thresholds, df, reference, metadata)
    kw[key] = value
    ok, failures = evaluate_release(**kw)
    assert not ok and any(fragment in f for f in failures)


def test_release_blocked_when_reference_population_violates_contract(contract, thresholds, df, reference, metadata):
    kw = _release_kwargs(contract, thresholds, df, reference, metadata)
    kw["reference_validation"] = run(df.assign(row_id=1), contract, reference, metadata)
    ok, failures = evaluate_release(**kw)
    assert not ok and any("poblacion de referencia" in f for f in failures)


def test_contract_and_thresholds_are_coherent(contract, thresholds):
    band, th = contract["validation"]["positivity"], thresholds["positivity"]
    assert (band["min_propensity"], band["max_propensity"]) == (th["warning_min"], th["warning_max"])
    assert th["hard_min"] < th["warning_min"] and th["hard_max"] > th["warning_max"]
