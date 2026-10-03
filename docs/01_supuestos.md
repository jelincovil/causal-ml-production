# 01. Supuestos de identificacion

Los supuestos de identificacion **no desaparecen al desplegar** el modelo. Este repositorio los declara,
los traduce a reglas ejecutables cuando es posible y dice con claridad cuales no se pueden verificar.

| Supuesto | Declarado en | Que se comprueba operacionalmente | Que NO se puede comprobar |
|---|---|---|---|
| **Pregunta causal y estimando** (ATE de `treatment` sobre `outcome`) | `contracts/causal_contract.yaml` (`causal`) | El estimando del contrato es soportado por el estimador (`validation/causal_checks.py`) | Que el ATE sea la cantidad que importa para la decision |
| **DAG / conjunto de ajuste** (25 covariables pre-tratamiento) | contrato + `causal_model/graph.py` | Criterio de puerta trasera sobre el DAG; ninguna covariable es descendiente del tratamiento; DoWhy identifica el estimando (release) | Que el DAG sea el verdadero |
| **Ignorabilidad condicional** | contrato | Nada: no es verificable. Se hace un analisis de sensibilidad (valor de robustez, ver `methodology.md`) y se reporta como `ROBUST` o `REVIEW`, nunca como `PASS` | Ausencia de confusion no observada |
| **Positividad / overlap** | `contracts/thresholds.yaml` (`positivity`) | `P(T=1 \| X)` con cross-fitting; minimo, maximo, observaciones extremas, ESS. Bloquea fuera de `[hard_min, hard_max]` | Positividad en regiones de X sin observaciones |
| **Consistencia** (definicion estable del tratamiento) | `versions` del contrato | Compatibilidad de version entre definicion del tratamiento, pipeline, DAG y modelo; valores de `treatment` permitidos; hash del artefacto | Que "tratado" signifique lo mismo en la practica |
| **SUTVA** (sin interferencia) | contrato (`validation.sutva`) | Con una variable de agrupacion: ICC del tratamiento y saturacion por cluster. Sin ella: `NOT PROVABLE` | Ausencia absoluta de interferencia |
| **Especificacion PLR** | `causal.estimator` | Prueba sintetica con efecto conocido (`validation/synthetic.py`); refutaciones DoWhy | Forma funcional correcta en datos reales |

La app **no** afirma "SUTVA verificada" ni "ignorabilidad = PASS". Afirma que ciertos diagnosticos no
encontraron senales incompatibles con los supuestos y que el resto requiere justificacion sustantiva.

## Donde se declaran en pantalla

La app muestra estos supuestos en la seccion **Supuestos del analisis**, organizada en tres niveles (verificados con
los datos, evidencia indirecta, no verificables) e incluyendo los supuestos del metodo: homogeneidad del efecto bajo
PLR (`X=None`), nuisances con LassoCV sin diagnostico de convergencia y normalidad asintotica del IC. El texto sale de
`validation/assumptions.py`, se escribe en `artifacts/model/model_card.json` y la interfaz lo renderiza tal cual; los
tests comprueban que card, catalogo, contrato y codigo no divergen.
