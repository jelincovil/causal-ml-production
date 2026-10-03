"""Los supuestos declarados deben coincidir entre contrato, codigo, model card y pantalla."""

import inspect
import json
import re
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from causal_model import estimator
from components.data_input import SCENARIOS
from contracts import ROOT
from validation.assumptions import (
    EMPIRICAL,
    INDIRECT,
    LEVELS,
    SCOPES,
    UNVERIFIABLE,
    build_assumptions,
    card_item,
    static_view,
)
from validation.model_card import validate_model_card
from validation.release_gate import ValidationResult, validate_runtime_data
from validation.results import INDIRECT_CHECKS, OPERATIONAL_CHECKS, REFUTATION, SKIPPED, WARN, CheckResult

# Entradas del contrato que son nombres de columnas o una semilla, no supuestos.
NOT_ASSUMPTIONS = {"causal.treatment", "causal.outcome", "causal.estimator.seed"}

CARD_PATH = ROOT / "artifacts" / "model" / "model_card.json"


@pytest.fixture(scope="module")
def catalog(contract):
    return build_assumptions(contract)


@pytest.fixture(scope="module")
def card():
    return json.loads(CARD_PATH.read_text(encoding="utf-8"))


def contract_units(contract):
    units = set()
    for section in ("causal", "versions", "validation"):
        for key, value in contract[section].items():
            if section == "causal" and key == "estimator":
                units |= {f"causal.estimator.{k}" for k in value}
            else:
                units.add(f"{section}.{key}")
    return units


def resolve(contract, dotted):
    node = contract
    for part in dotted.split("."):
        node = node[part]
    return node


# ---------------------------------------------------------------- catalogo y contrato


def test_catalog_is_well_formed(catalog):
    assert len({a.id for a in catalog}) == len(catalog)
    assert len({a.name for a in catalog}) == len(catalog)
    levels = {level for level, _, _ in LEVELS}
    live_rows = set(OPERATIONAL_CHECKS) | set(INDIRECT_CHECKS)
    for a in catalog:
        assert a.level in levels and a.scope in SCOPES
        assert a.function and a.description
        if a.level != UNVERIFIABLE:
            assert a.status_row in live_rows, f"{a.id}: sin fila de validacion que muestre su estado"
    assert {a.level for a in catalog} == levels


def test_every_contract_declaration_is_covered_by_an_assumption(contract, catalog):
    covered = {key for a in catalog for key in a.contract_keys}
    missing = contract_units(contract) - NOT_ASSUMPTIONS - covered
    assert not missing, f"entradas del contrato sin supuesto declarado: {sorted(missing)}"
    for key in covered:
        resolve(contract, key)  # falla con KeyError si el catalogo cita una entrada inexistente


def test_method_assumptions_are_declared_as_unverified(catalog):
    by_id = {a.id: a for a in catalog}
    for key in ("effect_homogeneity", "nuisance_estimation", "asymptotic_normality"):
        assert by_id[key].scope == "metodo" and by_id[key].level == UNVERIFIABLE, key
        assert by_id[key].status_row is None


# ------------------------------------------------- las declaraciones coinciden con el codigo


def test_declarations_match_the_implementation(contract, catalog):
    by_id = {a.id: a for a in catalog}
    folds = contract["causal"]["estimator"]["cv_folds"]

    assert "X=None" in inspect.getsource(estimator.fit_effect_model)
    assert "X=None" in by_id["effect_homogeneity"].description

    est = estimator.new_estimator(contract)
    assert type(est.model_y).__name__ == "LassoCV" and type(est.model_t).__name__ == "LassoCV"
    assert est.cv == folds
    assert "LassoCV" in by_id["nuisance_estimation"].description
    assert f"K={folds}" in by_id["nuisance_estimation"].description

    assert "ate_inference" in inspect.getsource(estimator.ate_summary)
    assert "ate_inference()" in by_id["asymptotic_normality"].description


def test_no_nuisance_diagnostics_exist_in_the_code():
    """Si alguien agrega un diagnostico de las nuisances, la declaracion 'sin diagnostico' debe actualizarse."""
    pattern = re.compile(r"r2_score|mean_squared_error|cross_val_score|nuisance_score|models_y|models_t")
    skip = {"assumptions.py", "test_assumptions.py"}
    files = [ROOT / "app.py"] + [
        p for d in ("causal_model", "validation", "monitoring", "components") for p in (ROOT / d).glob("*.py")
    ]
    offenders = [p.name for p in files if p.name not in skip and pattern.search(p.read_text(encoding="utf-8"))]
    assert not offenders, f"hay diagnosticos de nuisances en {offenders}: actualice validation/assumptions.py"


# ------------------------------------------------------------ model card = catalogo


def test_model_card_assumptions_equal_the_catalog(contract, catalog, card):
    assert [static_view(s) for s in card["supuestos"]] == [static_view(card_item(a)) for a in catalog]
    assert validate_model_card(card) == []


def test_model_card_verification_semantics(card):
    for s in card["supuestos"]:
        assert s["verificado"] == (s["nivel"] == EMPIRICAL), s["id"]
        if s["nivel"] in (EMPIRICAL, INDIRECT):
            assert s.get("evidencia"), s["id"]
        else:
            assert not s.get("evidencia"), s["id"]
    plr = next(s for s in card["supuestos"] if s["id"] == "effect_homogeneity")
    assert plr["verificado"] is False and plr["nivel"] == UNVERIFIABLE


# --------------------------------------------------------------- operacional vs indirecto


def test_every_check_row_belongs_to_exactly_one_group(df, contract, reference):
    import json as _json

    from causal_model.estimator import MODEL_DIR

    metadata = _json.loads((MODEL_DIR / "model_metadata.json").read_text(encoding="utf-8"))
    names = {c.name for c in validate_runtime_data(df, contract, reference, metadata).checks}
    names |= {REFUTATION}
    assert names <= set(OPERATIONAL_CHECKS) | set(INDIRECT_CHECKS)
    assert not set(OPERATIONAL_CHECKS) & set(INDIRECT_CHECKS)


def test_operational_status_excludes_refutation_but_global_status_does_not():
    v = ValidationResult(
        [CheckResult("Schema", "PASS"), CheckResult("Positivity", "PASS"), CheckResult(REFUTATION, WARN, "x")]
    )
    assert v.status == WARN
    assert v.operational_status == "PASS"


def test_expander_separates_operational_from_indirect():
    def script():
        from components.validation_status import render_validation_status
        from validation.release_gate import ValidationResult
        from validation.results import CheckResult

        render_validation_status(
            ValidationResult([CheckResult("Schema", "PASS"), CheckResult("Refutation (release)", "WARN", "x")])
        )

    at = AppTest.from_function(script, default_timeout=60).run()
    assert not at.exception
    text = next(m.value for m in at.markdown if "Operational checks" in m.value)
    assert "Operational checks** (se comprueban con los datos y los artefactos): :green[**PASS**]" in text
    assert "refutaciones :orange[**WARN**]" in text


# ------------------------------------------------------------------------- pantalla


def run_app(scenario=None):
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=120)
    at.run()
    if scenario:
        at.sidebar.selectbox(key="scenario").select(scenario)
        at.run()
    return at


def rendered_text(at):
    parts = []
    for kind in ("markdown", "caption", "subheader", "warning", "error", "success", "info"):
        parts += [e.value for e in getattr(at, kind)]
    return "\n".join(parts)


SCENARIOS_UNDER_TEST = [
    next(iter(SCENARIOS)),
    next(n for n in SCENARIOS if n.startswith("Esquema roto")),  # BLOCK: la seccion debe verse igual
]


@pytest.mark.parametrize("scenario", SCENARIOS_UNDER_TEST)
def test_each_assumption_appears_once_in_its_level(scenario, catalog):
    at = run_app(scenario)
    assert not at.exception
    text = rendered_text(at)
    for a in catalog:
        assert text.count(a.name) == 1, f"'{a.name}' aparece {text.count(a.name)} veces"

    markdown = [m.value for m in at.markdown]
    for level, title, _ in LEVELS:
        assert text.count(title) == 1
        table = markdown[next(i for i, v in enumerate(markdown) if title in v) + 1]
        for a in catalog:
            assert (a.name in table) == (a.level == level), f"{a.id} fuera de su nivel"


@pytest.mark.parametrize("scenario", SCENARIOS_UNDER_TEST)
def test_screen_text_is_identical_to_the_card_text(scenario, card):
    markdown = "\n".join(m.value for m in run_app(scenario).markdown)
    for s in card["supuestos"]:
        for field in ("nombre", "funcion", "descripcion"):
            assert s[field].replace("|", "\\|") in markdown, f"{s['id']}.{field} difiere de la card"


def test_every_contract_declaration_is_visible_on_screen(contract, catalog):
    text = rendered_text(run_app())
    for unit in sorted(contract_units(contract) - NOT_ASSUMPTIONS):
        names = [a.name for a in catalog if unit in a.contract_keys]
        assert names, unit
        assert all(name in text for name in names), unit


def test_unverifiable_assumptions_show_no_verdict(catalog):
    at = run_app()
    markdown = [m.value for m in at.markdown]
    title = next(t for lvl, t, _ in LEVELS if lvl == UNVERIFIABLE)
    table = markdown[next(i for i, v in enumerate(markdown) if title in v) + 1]
    rows = [r for r in table.splitlines()[2:]]
    n_unverifiable = sum(a.level == UNVERIFIABLE for a in catalog)
    assert len(rows) == n_unverifiable
    assert all("NO VERIFICADO" in r for r in rows)
    assert not any(re.search(r"\*\*(PASS|WARN|BLOCK|ROBUST)\*\*", r) for r in rows)


def test_method_assumptions_are_on_screen_as_not_verified():
    text = rendered_text(run_app())
    for fragment in ("X=None", "LassoCV", "ate_inference()", "diagnostico de la calidad ni de la convergencia"):
        assert fragment in text, fragment


def test_positivity_panel_states_its_specification():
    at = run_app()
    caption = next(c.value for c in at.caption if "El propensity mostrado" in c.value)
    for fragment in ("regresion logistica con cross-fitting", "distinta del LassoCV", "se juzga bajo esa especificacion"):
        assert fragment in caption, fragment


def test_reestimation_warns_that_identification_must_hold_in_the_input_population():
    at = run_app()
    assert any(
        "tambien se cumplen en esa poblacion" in w.value and "Ningun check de esta aplicacion lo garantiza" in w.value
        for w in at.warning
    )


@pytest.mark.parametrize("scenario", SCENARIOS_UNDER_TEST)
def test_versions_row_is_labelled_as_artifact_audit(scenario):
    text = rendered_text(run_app(scenario))
    assert "audita artefactos" in text and "no evalua la consistencia causal del tratamiento" in text


def test_panel_renders_only_what_the_card_says():
    def script():
        import json

        from components.assumptions_panel import render_assumptions
        from contracts import ROOT
        from validation.release_gate import ValidationResult

        card = json.loads((ROOT / "artifacts" / "model" / "model_card.json").read_text(encoding="utf-8"))
        card["supuestos"][0]["descripcion"] = "TEXTO_QUE_SOLO_ESTA_EN_LA_CARD"
        render_assumptions(card, ValidationResult(checks=[]))

    at = AppTest.from_function(script, default_timeout=60).run()
    assert not at.exception
    assert any("TEXTO_QUE_SOLO_ESTA_EN_LA_CARD" in m.value for m in at.markdown)


def test_verdicts_are_unchanged_by_the_informational_sections():
    at = run_app()
    assert not at.error
    assert any("EFFECT ESTIMATION: ALLOWED" in s.value for s in at.success)
    blocked = run_app(next(n for n in SCENARIOS if n.startswith("Esquema roto")))
    assert any("EFFECT ESTIMATION: BLOCKED" in e.value for e in blocked.error)
