"""Historial de monitoring.

Community Cloud no garantiza persistencia del disco local: el historial duradero exige una base
de datos o almacenamiento externo configurado con st.secrets. Sin eso, el historial vive solo en la sesion.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Mapping, MutableSequence, Optional

import pandas as pd

COLUMNS = ["ts", "model_version", "status", "n", "psi_max", "ks_max", "treatment_rate", "ate_runtime"]


def make_record(
    *, model_version: str, status: str, profile: Dict[str, Any], ate_runtime: Optional[float]
) -> Dict[str, Any]:
    return {
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model_version": model_version,
        "status": status,
        "n": profile["n"],
        "psi_max": profile["psi_max"],
        "ks_max": profile["ks_max"],
        "treatment_rate": profile["treatment_rate"],
        "ate_runtime": ate_runtime,
    }


class SessionHistory:
    persistent = False

    def __init__(self, store: MutableSequence[Dict[str, Any]]):
        self._store = store

    def append(self, record: Dict[str, Any]) -> None:
        self._store.append(record)

    def read(self) -> pd.DataFrame:
        return pd.DataFrame(list(self._store), columns=COLUMNS)


class SQLHistory:
    persistent = True

    def __init__(self, url: str):
        from sqlalchemy import create_engine, text

        self._text = text
        self._engine = create_engine(url)
        with self._engine.begin() as conn:
            conn.execute(
                text(
                    "CREATE TABLE IF NOT EXISTS monitoring_history ("
                    "ts TEXT, model_version TEXT, status TEXT, n INTEGER, "
                    "psi_max REAL, ks_max REAL, treatment_rate REAL, ate_runtime REAL)"
                )
            )

    def append(self, record: Dict[str, Any]) -> None:
        with self._engine.begin() as conn:
            conn.execute(
                self._text(
                    "INSERT INTO monitoring_history VALUES "
                    "(:ts, :model_version, :status, :n, :psi_max, :ks_max, :treatment_rate, :ate_runtime)"
                ),
                record,
            )

    def read(self) -> pd.DataFrame:
        with self._engine.connect() as conn:
            return pd.read_sql(self._text("SELECT * FROM monitoring_history ORDER BY ts"), conn)


def history_store(secrets: Mapping[str, Any], session_list: MutableSequence[Dict[str, Any]]):
    """SQLHistory si hay [history].url en st.secrets; en caso contrario SessionHistory."""
    try:
        url = secrets["history"]["url"]
    except (KeyError, FileNotFoundError):
        return SessionHistory(session_list)
    return SQLHistory(url)
