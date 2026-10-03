"""Catalogo unico de supuestos: que se verifica con los datos, que tiene evidencia indirecta y que no es verificable.

Es la unica fuente del texto: validate_release.py lo escribe en la seccion `supuestos` de
artifacts/model/model_card.json y la app renderiza esa seccion. No hay otro lugar donde se redacten estos supuestos.
Todo es informativo: no cambia ningun veredicto PASS/WARN/BLOCK ni los umbrales.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from validation.results import (
    DOMAIN,
    DRIFT,
    MISSING,
    POSITIVITY,
    REFUTATION,
    SCHEMA,
    SENSITIVITY,
    SUTVA,
    TREATMENT,
)

EMPIRICAL = "empirico"
INDIRECT = "indirecto"
UNVERIFIABLE = "no_verificable"

LEVELS: Tuple[Tuple[str, str, str], ...] = (
    (
        EMPIRICAL,
        "Nivel 1. Verificados empiricamente con los datos de entrada",
        "Cada uno se contrasta con una funcion y un umbral en esta ejecucion; el estado es el de esa verificacion.",
    ),
    (
        INDIRECT,
        "Nivel 2. Evaluados con evidencia indirecta",
        "No se pueden comprobar; hay una medida que los hace mas o menos plausibles. Nunca devuelven PASS por si solos.",
    ),
    (
        UNVERIFIABLE,
        "Nivel 3. No verificables por la aplicacion",
        "La aplicacion los supone o los declara pero no los comprueba: requieren justificacion sustantiva.",
    ),
)

SCOPES = ("datos", "identificacion", "metodo")

# Campos que la model card y el catalogo deben tener identicos (la evidencia del release no entra aqui).
STATIC_FIELDS = (
    "id",
    "nombre",
    "nivel",
    "ambito",
    "funcion",
    "descripcion",
    "fila_validacion",
    "declarado_en",
    "verificado",
)


@dataclass(frozen=True)
class Assumption:
    id: str
    name: str
    level: str
    scope: str
    function: str
    description: str
    status_row: Optional[str] = None  # fila de validate_runtime_data cuyo estado se muestra (niveles 1 y 2)
    contract_keys: Tuple[str, ...] = ()  # entradas del contrato donde esta declarado


def build_assumptions(contract: Dict[str, Any]) -> List[Assumption]:
    folds = contract["causal"]["estimator"]["cv_folds"]
    cluster = contract["validation"]["sutva"].get("cluster_column")

    if cluster:
        sutva = Assumption(
            "sutva",
            "SUTVA (sin interferencia)",
            INDIRECT,
            "identificacion",
            "validation.sutva_checks.check_sutva",
            f"check_sutva reporta el ICC del tratamiento y la saturacion por cluster ({cluster}) como senal indirecta "
            "de derrames entre unidades; no prueba la ausencia de interferencia.",
            SUTVA,
            ("validation.sutva",),
        )
    else:
        sutva = Assumption(
            "sutva",
            "SUTVA (sin interferencia)",
            UNVERIFIABLE,
            "identificacion",
            "validation.sutva_checks.check_sutva",
            "No es verificable con estos datos. check_sutva solo reporta ICC y saturacion por cluster si el contrato "
            "define cluster_column (hoy: null); sin esa variable no evalua nada y la ausencia de interferencia "
            "requiere justificacion sustantiva.",
            SUTVA,
            ("validation.sutva",),
        )

    return [
        # Nivel 1: con los datos de entrada
        Assumption(
            "schema",
            "Esquema de entrada conforme al contrato",
            EMPIRICAL,
            "datos",
            "validation.schema_checks.check_schema",
            "Que las columnas, los tipos y los nulos de la entrada coinciden con feature_schema.json; "
            "si no, el estimador no recibe lo que se entreno.",
            SCHEMA,
            ("validation.schema",),
        ),
        Assumption(
            "treatment_coding",
            "Tratamiento binario con ambos niveles",
            EMPIRICAL,
            "datos",
            "validation.schema_checks.check_treatment",
            "Que treatment solo toma los valores permitidos por el contrato y aparecen ambos niveles; "
            "sin contraste entre tratados y controles no hay efecto que estimar.",
            TREATMENT,
            ("causal.treatment_levels", "validation.treatment"),
        ),
        Assumption(
            "positivity",
            "Positividad (overlap)",
            EMPIRICAL,
            "identificacion",
            "validation.positivity_checks.check_positivity",
            "Que ninguna unidad tiene probabilidad de tratamiento cercana a 0 o a 1. Se juzga con el propensity de "
            "una regresion logistica con cross-fitting, distinta del LassoCV que usa el estimador: es una verificacion "
            "bajo esa especificacion, no sobre la asignacion real. Umbrales en thresholds.yaml.",
            POSITIVITY,
            ("validation.positivity",),
        ),
        Assumption(
            "domain",
            "Entrada dentro del dominio de referencia",
            EMPIRICAL,
            "datos",
            "monitoring.monitoring_report.check_domain",
            "Que cada covariable de la entrada cae dentro del rango [q01, q99] observado en el entrenamiento; "
            "fuera de ese rango el modelo extrapola.",
            DOMAIN,
            ("validation.domain",),
        ),
        Assumption(
            "drift",
            "Poblacion de entrada estable respecto de la referencia",
            EMPIRICAL,
            "datos",
            "monitoring.monitoring_report.check_drift",
            "Que la distribucion de cada covariable (PSI) no se aleja de la poblacion de referencia; "
            "una poblacion distinta puede invalidar el efecto aprendido.",
            DRIFT,
            ("validation.drift",),
        ),
        Assumption(
            "missingness",
            "Datos suficientemente completos",
            EMPIRICAL,
            "datos",
            "monitoring.monitoring_report.check_missingness",
            "Que la tasa de nulos en las covariables esta bajo el umbral; la estimacion usa solo los casos completos.",
            MISSING,
            ("validation.missingness",),
        ),
        # Nivel 2: evidencia indirecta
        Assumption(
            "unobserved_confounding",
            "Ignorabilidad condicional (sin confusion no observada)",
            INDIRECT,
            "identificacion",
            "validation.sensitivity_checks.check_sensitivity",
            "No es verificable: se mide cuanta confusion no observada (valor de robustez de Cinelli y Hazlett) haria "
            "falta para anular el efecto. El resultado es ROBUST o REVIEW, nunca PASS, y no prueba que no haya confusion.",
            SENSITIVITY,
            ("causal.pre_treatment_covariates", "validation.sensitivity"),
        ),
        Assumption(
            "refutation",
            "Efecto no espurio y estable (refutaciones)",
            INDIRECT,
            "metodo",
            "causal_model.refutation.run_refutations",
            "Evidencia del release: con un tratamiento placebo el efecto desaparece y una causa comun aleatoria o un "
            "subconjunto de datos no lo cambian. Se calcula sobre la poblacion de entrenamiento y no se recalcula con "
            "la entrada.",
            REFUTATION,
            ("validation.refutation",),
        ),
        # Nivel 3: no verificables (identificacion y datos)
        Assumption(
            "dag",
            "Estructura causal (DAG) correcta",
            UNVERIFIABLE,
            "identificacion",
            "validation.causal_checks.check_graph_contract",
            "Los datos no pueden confirmar el DAG. La funcion solo audita su coherencia interna: covariables del "
            "contrato, schema y nodos coinciden, y el ajuste cumple el criterio de puerta trasera sobre el grafo declarado.",
            None,
            ("causal.estimand",),
        ),
        sutva,
        Assumption(
            "consistency",
            "Consistencia (definicion estable del tratamiento)",
            UNVERIFIABLE,
            "identificacion",
            "ninguna",
            "Que 'tratado' significa lo mismo en el entrenamiento y en la entrada. Ninguna funcion lo evalua: la fila "
            "Versions & integrity (check_versions) audita artefactos (versiones, hashes y metadatos del modelo), "
            "no la consistencia causal del tratamiento.",
            None,
            ("versions.treatment_definition", "versions.preprocessing", "versions.causal_graph"),
        ),
        Assumption(
            "identification_in_input",
            "Identificacion valida en la poblacion de entrada",
            UNVERIFIABLE,
            "identificacion",
            "ninguna",
            "El ATE re-estimado sobre la entrada supone que los supuestos de identificacion tambien valen en esa "
            "poblacion. Ningun check lo garantiza: dominio y drift solo detectan diferencias observables en las covariables.",
            None,
        ),
        Assumption(
            "missing_at_random",
            "Datos faltantes ignorables (casos completos)",
            UNVERIFIABLE,
            "datos",
            "ninguna",
            "Que descartar las filas con nulos (casos completos) no sesga el efecto. check_missingness solo mide la "
            "tasa de nulos; no contrasta el mecanismo de los datos faltantes.",
            None,
        ),
        # Nivel 3: supuestos del metodo
        Assumption(
            "effect_homogeneity",
            "Homogeneidad del efecto (PLR con X=None)",
            UNVERIFIABLE,
            "metodo",
            "causal_model.estimator.fit_effect_model",
            "El estimador se ajusta con X=None (especificacion parcialmente lineal): impone un unico efecto para todas "
            "las unidades. La app no contrasta esa homogeneidad; si el efecto es heterogeneo, el ATE es un promedio "
            "que no describe a subgrupos ni a individuos.",
            None,
            ("causal.estimator.method",),
        ),
        Assumption(
            "nuisance_estimation",
            "Nuisances bien estimadas (LassoCV, sin diagnostico de convergencia)",
            UNVERIFIABLE,
            "metodo",
            "causal_model.estimator.new_estimator",
            f"E[Y|X] y E[T|X] se estiman con LassoCV, con cross-fitting de K={folds} folds. La app no calcula ningun "
            "diagnostico de la calidad ni de la convergencia de estas nuisances (R2 fuera de muestra, estabilidad "
            "entre folds, tasas de error).",
            None,
            ("causal.estimator.model_y", "causal.estimator.model_t", "causal.estimator.cv_folds"),
        ),
        Assumption(
            "asymptotic_normality",
            "Normalidad asintotica del estimador (IC 95%)",
            UNVERIFIABLE,
            "metodo",
            "causal_model.estimator.ate_summary",
            "El IC 95% usa el error estandar y la aproximacion normal de ate_inference() de EconML. La app no verifica "
            "que la muestra sea suficiente para esa aproximacion ni la cobertura real del intervalo.",
            None,
        ),
    ]


def card_item(a: Assumption) -> Dict[str, Any]:
    """Parte estatica de un supuesto tal como se escribe en la model card."""
    return {
        "id": a.id,
        "nombre": a.name,
        "nivel": a.level,
        "ambito": a.scope,
        "funcion": a.function,
        "descripcion": a.description,
        "fila_validacion": a.status_row,
        "declarado_en": list(a.contract_keys),
        "verificado": a.level == EMPIRICAL,
    }


def static_view(item: Dict[str, Any]) -> Dict[str, Any]:
    return {k: item.get(k) for k in STATIC_FIELDS}
