#!/usr/bin/env python3
"""Genera artifacts/monitoring/reference_profile.json desde la poblacion de entrenamiento."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from causal_model.estimator import file_sha256  # noqa: E402
from contracts import DATASET_PATH, load_contract, load_dataset  # noqa: E402
from monitoring.reference_profile import PROFILE_PATH, build_reference_profile, save_reference_profile  # noqa: E402

DATA_VERSION = "ihdp_medium-train-v1"


def main() -> int:
    profile = build_reference_profile(
        load_dataset(), load_contract(), data_version=DATA_VERSION, data_sha256=file_sha256(DATASET_PATH)
    )
    save_reference_profile(profile)
    print(f"perfil de referencia guardado en {PROFILE_PATH} (n={profile['n']}, tasa de tratamiento={profile['treatment_rate']:.3f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
