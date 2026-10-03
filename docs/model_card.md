# Model card causal: `causal_dml` v1.0.0

> Version legible. La version machine-readable (`causal-model-card-v1`) esta en
> `artifacts/model/model_card.json` y se valida contra `contracts/causal-model-card-v1.schema.json`.

**Uso previsto: educativo / profesional, sin uso clinico ni de politica publica real.** Entrena sobre un
benchmark sintetico (IHDP-style) para practicar el flujo estimar, validar, desplegar y monitorear un
sistema causal.

## 1. Estimando

| Campo | Valor |
|---|---|
| Estimando | ATE = E[Y(1) - Y(0)] |
| Modelo | PLR / DML, `econml.dml.LinearDML`, K = 5, LassoCV (DML2) |
| Tratamiento / outcome | `treatment` (binario) / `outcome` (continuo) |
| Covariables | `x0` ... `x24`, todas pre-tratamiento |
| Poblacion de entrenamiento | `data/dataset.csv`, n = 1200 (split train de `synthetic_inference_ihdp_medium`) |
| ATE | 3.527 (SE 0.326), IC 95% [2.888, 4.166] |

## 2. Supuestos declarados

Ignorabilidad condicional, positividad, SUTVA y especificacion PLR. Detalle y estado de verificacion en
`docs/01_supuestos.md`. La ignorabilidad es cierta **por construccion del simulador**; no esta demostrada
para ningun dato real.

## 3. Evidencia de validacion (`artifacts/validation/`)

| Evidencia | Resultado |
|---|---|
| Identificacion (DoWhy) | backdoor, ajuste por las 25 covariables |
| Placebo / causa comun aleatoria / subconjunto de datos (50 simulaciones) | PASS / PASS / PASS (p = 0.48 / 0.35 / 0.29) |
| Valor de robustez | 27.0% (22.7% para IC con 0): `ROBUST` segun el criterio del proyecto |
| Prueba sintetica (efecto verdadero 2.0) | estimado 1.97, error absoluto 0.03 |
| Positividad en referencia | propensity en [0.11, 0.72] |
| Efecto verdadero del simulador (solo evaluacion) | ATE = 3.97; error absoluto de la estimacion ~0.44 |

El IC 95% de entrenamiento contiene el ATE verdadero del simulador (3.97), pero la estimacion puntual lo subestima en
~0.44 (alrededor de 1.4 errores estandar). Solo es posible compararlos porque el simulador revela el efecto verdadero;
en datos reales ese contraste no existe.

## 4. No usar para

- Decisiones clinicas o de politica publica reales.
- Poblaciones distintas de la de entrenamiento sin revisar el reporte de monitoring (la app avisa o bloquea).
- Decisiones individuales: es un ATE (efecto homogeneo bajo PLR), no un CATE.
- Presentar "validez causal = PASS": solo se verifican condiciones operacionales.

## 5. Mantenimiento

Cambie esta card en el mismo pull request que modifique el dataset, el contrato, el DAG o el estimador.
`scripts/validate_release.py` la regenera y la valida.
