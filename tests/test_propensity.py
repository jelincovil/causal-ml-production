import numpy as np

from causal_model.propensity import crossfit_propensity, propensity_summary


def test_reference_overlap(df, contract):
    covs = contract["causal"]["pre_treatment_covariates"]
    e = crossfit_propensity(df[covs].to_numpy(float), df["treatment"].to_numpy(int))
    assert 0 < e.min() and e.max() < 1
    assert 0.05 < e.min() and e.max() < 0.95


def test_summary_flags_extremes():
    rng = np.random.default_rng(0)
    e = rng.uniform(0.001, 0.999, 500)
    t = rng.binomial(1, e)
    s = propensity_summary(e, t, warning_min=0.05, warning_max=0.95)
    assert s["extreme_share"] > 0.05
    assert 0 < s["ess_ratio_treated"] <= 1
