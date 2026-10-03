"""El gate de release debe fallar (exit 1) y hacerlo rapido ante datos o artefactos inconsistentes."""

import shutil
import subprocess
import sys

from contracts import ROOT

IGNORE = shutil.ignore_patterns("__pycache__", ".pytest_cache", ".git", ".venv", "tests")


def run_release_in_copy(tmp_path, mutate):
    repo = tmp_path / "repo"
    shutil.copytree(ROOT, repo, ignore=IGNORE)
    mutate(repo)
    before = (repo / "artifacts" / "validation" / "validation_report.json").read_bytes()
    result = subprocess.run(
        [sys.executable, "-W", "ignore", "scripts/validate_release.py"],
        cwd=repo, capture_output=True, text=True, timeout=120,
    )
    after = (repo / "artifacts" / "validation" / "validation_report.json").read_bytes()
    return result, before == after


def test_non_numeric_dataset_fails_fast_without_overwriting_evidence(tmp_path):
    def mutate(repo):
        path = repo / "data" / "dataset.csv"
        lines = path.read_text().splitlines()
        lines[1] = "abc" + lines[1]
        path.write_text("\n".join(lines) + "\n")

    result, evidence_untouched = run_release_in_copy(tmp_path, mutate)
    assert result.returncode == 1
    assert "RELEASE BLOQUEADO" in result.stdout and "Schema" in result.stdout
    assert "[4/6]" not in result.stdout  # no se ejecutaron las refutaciones
    assert evidence_untouched


def test_modified_dataset_is_detected_by_hash(tmp_path):
    def mutate(repo):
        path = repo / "data" / "dataset.csv"
        path.write_text(path.read_text().replace("\n", "\n", 1) + path.read_text().splitlines()[1] + "\n")

    result, evidence_untouched = run_release_in_copy(tmp_path, mutate)
    assert result.returncode == 1
    assert "no es el dataset con el que se entreno el modelo" in result.stdout
    assert evidence_untouched
