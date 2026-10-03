# Metodologia

## Estimador

`causal_model/estimator.py` usa `econml.dml.LinearDML` con `X=None` y las 25 covariables como controles `W`:
una regresion parcialmente lineal (PLR) con *double machine learning*.

- Nuisances: `LassoCV` para `E[Y|X]` y para `E[T|X]` (el tratamiento binario se residualiza como continuo,
  igual que en la Serie A del curso).
- Cross-fitting con K = 5 folds; la etapa final usa todos los residuales fuera de muestra (variante DML2).
- Resultado en la poblacion de referencia: **ATE = 3.527**, SE = 0.326, IC 95% = [2.888, 4.166], n = 1200.
  Coincide con el PLR de la Serie A (3.528; IC [2.886, 4.169]).
- Bajo PLR el efecto es **homogeneo**: no hay CATE individual. Para focalizar con presupuesto limitado
  se necesita un estimador de efectos heterogeneos y otra validacion.

## Dos estimaciones en la app

`estimate_effect` devuelve siempre dos cosas, que no deben confundirse:

1. **Referencia:** el ATE del modelo desplegado (`causal_model.joblib`), entrenado con `data/dataset.csv`.
2. **Entrada:** el mismo estimador reajustado sobre los casos completos de los datos de entrada. Solo se
   ejecuta si `validate_runtime_data` no produjo `BLOCK`. La diferencia entre ambas es informacion, no una prueba.

## Positividad

`causal_model/propensity.py`: regresion logistica (C = 1) con cross-fitting estratificado de 5 folds.
Se reportan minimo, maximo, cuantiles 1% y 99%, fraccion de observaciones fuera de la banda y ESS/n de los
pesos IPW de tratados y controles.

## Refutaciones (DoWhy 0.14)

`causal_model/refutation.py` identifica el estimando con el DAG (backdoor) y estima con
`backdoor.econml.dml.LinearDML`. Refutadores: `placebo_treatment_refuter` (permutacion),
`random_common_cause` y `data_subset_refuter` (80%). Con 50 simulaciones por refutador, un refutador
**pasa** si su p-valor supera 0.05. El resultado se guarda en `artifacts/validation/refutation_report.json`;
la app solo lo presenta (no se recalcula con la entrada actual).

## Sensibilidad a confusion no observada

`validation/sensitivity_checks.py` calcula el **valor de robustez** de Cinelli y Hazlett (2020) sobre la
regresion de residuales (`t = ATE/SE`, `dof = n - p - 2`): la fuerza minima (R2 parcial con `T` y con `Y`)
que un confusor no observado necesitaria para anular la estimacion. En la referencia: 27.0% (22.7% para
que el IC incluya 0). Es una aproximacion aplicada a los residuales del DML, no una prueba de ignorabilidad.

## Monitoring

PSI con deciles de referencia (proporciones recortadas a 1e-4); KS de una muestra contra la CDF de referencia
reconstruida con 101 cuantiles; fraccion fuera de `[q01, q99]`; tasa de tratamiento y de nulos.

## Limites

- Datos sinteticos: la validez externa no esta en juego, solo la mecanica del sistema.
- Que los checks pasen no demuestra validez causal: ignorabilidad verdadera, ausencia absoluta de
  interferencia y validez causal universal no son demostrables por una app.
- El efecto se estima para el estimando ATE de la poblacion de entrenamiento; no es un efecto garantizado
  para una unidad individual.
