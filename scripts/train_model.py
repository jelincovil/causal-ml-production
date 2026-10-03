#!/usr/bin/env python3
"""Entrena el estimador del contrato sobre data/dataset.csv y escribe artifacts/model/."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from causal_model.estimator import MODEL_DIR, build_metadata, file_sha256, fit_effect_model, save_bundle  # noqa: E402
from contracts import DATASET_PATH, load_contract, load_dataset  # noqa: E402


def main() -> int:
    contract = load_contract()
    est, n = fit_effect_model(load_dataset(), contract)
    metadata = build_metadata(est, contract, n_train=n, data_sha256=file_sha256(DATASET_PATH))
    saved = save_bundle(est, metadata)
    e = saved["effect"]
    print(f"modelo guardado en {MODEL_DIR}")
    print(f"ATE = {e['ate']:.4f}  SE = {e['se']:.4f}  IC95 = [{e['ci_low']:.4f}, {e['ci_high']:.4f}]  n = {n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
