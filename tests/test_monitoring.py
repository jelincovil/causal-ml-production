import numpy as np
import pandas as pd
import pytest

from monitoring.drift import feature_drift_table, ks_vs_reference, psi
from monitoring.history import SQLHistory, SessionHistory, history_store, make_record
from monitoring.reference_profile import build_reference_profile, load_reference_profile, save_reference_profile
from monitoring.runtime_profile import build_runtime_profile


def test_psi_is_small_for_same_population(df, reference):
    table = feature_drift_table(df, reference)
    assert table["psi"].max() < 0.02
    assert table["out_of_range_rate"].max() < 0.03


def test_psi_and_ks_detect_shift(df, reference):
    shifted = df.assign(x0=df["x0"] + 1.0)
    ref = reference["features"]["x0"]
    assert psi(shifted["x0"].to_numpy(), ref) > 0.25
    assert ks_vs_reference(shifted["x0"].to_numpy(), ref) > 0.3
    assert ks_vs_reference(df["x0"].to_numpy(), ref) < 0.03


def test_runtime_profile_reports_worst_feature(df, contract, reference):
    p = build_runtime_profile(df.assign(x4=df["x4"] * 3), contract, reference)
    assert p["psi_max_feature"] == "x4"
    assert p["treatment_rate"] == pytest.approx(reference["treatment_rate"])


def test_reference_profile_roundtrip(tmp_path, df, contract):
    profile = build_reference_profile(df, contract, data_version="t", data_sha256="0" * 64)
    save_reference_profile(profile, tmp_path / "p.json")
    assert load_reference_profile(tmp_path / "p.json")["n"] == len(df)


def _record():
    profile = {"n": 10, "psi_max": 0.1, "ks_max": 0.2, "treatment_rate": 0.4}
    return make_record(model_version="1.0.0", status="PASS", profile=profile, ate_runtime=3.5)


def test_session_history():
    h = SessionHistory([])
    assert h.read().empty and not h.persistent
    h.append(_record())
    assert len(h.read()) == 1


def test_sql_history_persists_across_instances(tmp_path):
    url = f"sqlite:///{tmp_path / 'h.db'}"
    SQLHistory(url).append(_record())
    reopened = SQLHistory(url)
    assert reopened.persistent and len(reopened.read()) == 1


def test_history_store_selection(tmp_path):
    assert isinstance(history_store({}, []), SessionHistory)
    assert isinstance(history_store({"history": {"url": f"sqlite:///{tmp_path / 'h.db'}"}}, []), SQLHistory)
