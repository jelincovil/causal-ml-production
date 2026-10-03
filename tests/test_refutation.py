from causal_model.graph import build_dag
from causal_model.refutation import run_refutations


def test_refutation_report_structure(df, contract, thresholds):
    report = run_refutations(df, contract, build_dag(contract), thresholds, num_simulations=3)
    assert report["model_version"] == contract["model"]["version"]
    assert set(report["refutations"]) == {"placebo_treatment", "random_common_cause", "data_subset"}
    for r in report["refutations"].values():
        assert r["status"] in {"PASS", "FAIL"}
        assert 0.0 <= r["p_value"] <= 1.0
    placebo = report["refutations"]["placebo_treatment"]
    assert abs(placebo["refuted_effect"]) < abs(placebo["original_effect"]) / 5
