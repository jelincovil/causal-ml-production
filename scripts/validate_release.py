#!/usr/bin/env python3
"""Genera la evidencia de validez del release. Sale con codigo 1 si falta una condicion obligatoria."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from causal_model.estimator import MODEL_DIR, MODEL_FILE, estimate_effect, file_sha256, load_bundle  # noqa: E402
from causal_model.graph import build_dag  # noqa: E402
from causal_model.identification import identify_with_dowhy  # noqa: E402
from causal_model.refutation import run_refutations  # noqa: E402
from contracts import ROOT, DATASET_PATH, load_contract, load_dataset, load_feature_schema, load_thresholds  # noqa: E402
from monitoring.reference_profile import load_reference_profile  # noqa: E402
from validation.model_card import build_model_card, validate_model_card  # noqa: E402
from validation.release_gate import evaluate_release, validate_runtime_data  # noqa: E402
from validation.results import BLOCK  # noqa: E402
from validation.sensitivity_checks import check_sensitivity  # noqa: E402
from validation.synthetic import run_synthetic_recovery  # noqa: E402

VALIDATION_DIR = ROOT / "artifacts" / "validation"


def _coherence_problems(contract, thresholds, metadata, profile) -> list:
    problems = []
    band, th = contract["validation"]["positivity"], thresholds["positivity"]
    if (band["min_propensity"], band["max_propensity"]) != (th["warning_min"], th["warning_max"]):
        problems.append("la banda de positividad del contrato no coincide con thresholds.yaml")
    sha = file_sha256(DATASET_PATH)
    if metadata["training_data_sha256"] != sha:
        problems.append("data/dataset.csv no es el dataset con el que se entreno el modelo (reentrenar)")
    if profile["training_data_sha256"] != sha:
        problems.append("reference_profile.json no corresponde a data/dataset.csv (regenerar)")
    if profile["model_version"] != contract["model"]["version"]:
        problems.append("reference_profile.json corresponde a otra version del modelo")
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-simulations", type=int, default=None, help="simulaciones por refutador")
    args = parser.parse_args()

    contract, thresholds, schema = load_contract(), load_thresholds(), load_feature_schema()
    df, graph = load_dataset(), build_dag(contract)
    bundle, profile = load_bundle(), load_reference_profile()

    print("[1/6] consistencia de artefactos")
    problems = _coherence_problems(contract, thresholds, bundle.metadata, profile)

    print("[2/6] contrato sobre la poblacion de referencia")
    reference = validate_runtime_data(
        df, contract, profile, bundle.metadata, thresholds=thresholds, feature_schema=schema,
        graph=graph, model_path=MODEL_DIR / MODEL_FILE,
    )
    if reference.has_blocking_error:
        problems.append(
            f"la poblacion de referencia incumple el contrato: {[c.name for c in reference.checks if c.status == BLOCK]}"
        )
    if problems:
        # Falla rapido: con datos o artefactos inconsistentes los pasos caros (DoWhy) no tienen sentido
        # y no se sobrescribe la evidencia anterior.
        print("-" * 60)
        print("RELEASE BLOQUEADO (exit 1):")
        for f in problems:
            print(f"  - {f}")
        return 1

    print("[3/6] identificacion (DoWhy)")
    identification = identify_with_dowhy(df, contract, graph)

    print("[4/6] refutaciones (DoWhy)")
    refutation = run_refutations(df, contract, graph, thresholds, num_simulations=args.num_simulations)

    print("[5/6] sensibilidad a confusion no observada")
    runtime = estimate_effect(bundle, df, contract)["runtime"]
    sensitivity = check_sensitivity(runtime, thresholds)

    print("[6/6] prueba causal sintetica y model card")
    synthetic = run_synthetic_recovery(contract, thresholds)
    card = build_model_card(bundle.metadata, refutation, contract, reference, sensitivity)
    card_errors = validate_model_card(card)

    ok, failures = evaluate_release(
        contract=contract, thresholds=thresholds, identification=identification,
        reference_validation=reference, refutation=refutation, sensitivity=sensitivity,
        synthetic=synthetic, consistency_problems=problems, model_card_errors=card_errors,
    )

    VALIDATION_DIR.mkdir(parents=True, exist_ok=True)
    (VALIDATION_DIR / "refutation_report.json").write_text(json.dumps(refutation, indent=2) + "\n", encoding="utf-8")
    report = {
        "model_version": contract["model"]["version"],
        "ok": ok,
        "failures": failures,
        "identification": identification,
        "reference_validation": reference.to_dict(),
        "sensitivity": {"status": sensitivity.status, "detail": sensitivity.detail, "metrics": sensitivity.metrics},
        "synthetic": synthetic,
    }
    (VALIDATION_DIR / "validation_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if not card_errors:
        (MODEL_DIR / "model_card.json").write_text(json.dumps(card, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print("-" * 60)
    for c in reference.checks:
        print(f"  {c.status:<8} {c.name}")
    for name, r in refutation["refutations"].items():
        print(f"  {r['status']:<8} refutacion: {name} (p={r['p_value']:.3f})")
    print(f"  {sensitivity.status:<8} {sensitivity.name}")
    print(f"  {'PASS' if synthetic['passed'] else 'FAIL':<8} prueba sintetica (error {synthetic['abs_error']:.3f})")
    print("-" * 60)
    if ok:
        print("RELEASE VALIDO (exit 0)")
        return 0
    print("RELEASE BLOQUEADO (exit 1):")
    for f in failures:
        print(f"  - {f}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
