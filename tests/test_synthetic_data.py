import pytest

from causal_model.estimator import ate_summary, fit_effect_model
from validation.synthetic import TRUE_ATE, run_synthetic_recovery, simulate


def test_estimator_recovers_known_effect(contract, thresholds):
    result = run_synthetic_recovery(contract, thresholds)
    assert result["passed"]
    assert result["abs_error"] < 0.15


def test_confounding_biases_the_naive_contrast(contract):
    d = simulate(contract)
    naive = d[d.treatment == 1].outcome.mean() - d[d.treatment == 0].outcome.mean()
    est, _ = fit_effect_model(d, contract)
    assert abs(ate_summary(est)["ate"] - TRUE_ATE) < abs(naive - TRUE_ATE)


def test_simulation_is_reproducible(contract):
    assert simulate(contract, seed=1).equals(simulate(contract, seed=1))
    assert not simulate(contract, seed=1).equals(simulate(contract, seed=2))
