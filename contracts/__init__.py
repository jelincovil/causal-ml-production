"""Carga de los contratos ejecutables versionados en este directorio."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

ROOT = Path(__file__).resolve().parents[1]
CONTRACTS_DIR = ROOT / "contracts"


def _yaml(name: str, path: Optional[Path]) -> Dict[str, Any]:
    with open(path or CONTRACTS_DIR / name, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def load_contract(path: Optional[Path] = None) -> Dict[str, Any]:
    return _yaml("causal_contract.yaml", path)


def load_thresholds(path: Optional[Path] = None) -> Dict[str, Any]:
    return _yaml("thresholds.yaml", path)


def load_feature_schema(path: Optional[Path] = None) -> Dict[str, Any]:
    with open(path or CONTRACTS_DIR / "feature_schema.json", encoding="utf-8") as fh:
        return json.load(fh)


DATASET_PATH = ROOT / "data" / "dataset.csv"
MODEL_CARD_SCHEMA_PATH = CONTRACTS_DIR / "causal-model-card-v1.schema.json"


def load_dataset(path: Optional[Path] = None):
    import pandas as pd

    return pd.read_csv(path or DATASET_PATH)
