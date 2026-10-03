import numpy as np
import pytest

from causal_model.estimator import (
    ModelBundle,
    ate_summary,
    build_metadata,
    complete_cases,
    estimate_effect,
    load_bundle,
    save_bundle,
)

REFERENCE_ATE = 3.527  # A2 del curso: PLR, K=5, LassoCV


def test_ate_matches_course_reference(fitted):
    est, n = fitted
    s = ate_summary(est)
    assert n == 1200
    assert s["ate"] == pytest.approx(REFERENCE_ATE, abs=0.05)
    assert s["ci_low"] < s["ate"] < s["ci_high"]
    assert s["se"] > 0


def test_complete_cases_drops_null_rows(df, contract):
    d = df.copy()
    d.loc[:9, "x5"] = np.nan
    assert len(complete_cases(d, contract)) == len(df) - 10


def test_save_load_roundtrip(tmp_path, fitted, contract):
    est, n = fitted
    meta = build_metadata(est, contract, n_train=n, data_sha256="0" * 64)
    saved = save_bundle(est, meta, tmp_path)
    bundle = load_bundle(tmp_path)
    assert bundle.metadata["model_sha256"] == saved["model_sha256"]
    assert ate_summary(bundle.estimator)["ate"] == pytest.approx(ate_summary(est)["ate"])


def test_estimate_effect_reference_and_runtime(fitted, df, contract):
    est, n = fitted
    meta = build_metadata(est, contract, n_train=n, data_sha256="0" * 64)
    out = estimate_effect(ModelBundle(est, meta), df, contract)
    assert out["reference"]["ate"] == pytest.approx(out["runtime"]["ate"], abs=1e-6)
    assert out["runtime"]["n"] == len(df) and out["n_dropped"] == 0
    assert out["runtime"]["dof"] == len(df) - 25 - 2
    assert out["model_version"] == contract["model"]["version"]
