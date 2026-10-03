# 02. Traduccion operacional

```text
SUPUESTO -> CONTRATO -> CHECK -> GATE -> EVIDENCIA -> MONITORING -> DECISION DE EJECUCION
```

| Capa | Archivo | Rol |
|---|---|---|
| Contrato | `contracts/causal_contract.yaml`, `feature_schema.json`, `thresholds.yaml` | que debe ser verdad para usar el modelo |
| Checks | `validation/*_checks.py`, `monitoring/` | comprobacion ejecutable |
| Gate | `validation/release_gate.py` | `PASS` / `WARN` / `BLOCK` en runtime y en release |
| Evidencia | `artifacts/validation/`, `artifacts/model/model_card.json` | resultados versionados |
| Monitoring | `artifacts/monitoring/reference_profile.json` | poblacion de referencia |

## Que bloquea y que advierte

| Condicion | Resultado | Fuente de la regla |
|---|---|---|
| Columnas desconocidas o faltantes, tipos no numericos, nulos en `treatment`/`outcome` | `BLOCK` | `validation.schema` del contrato |
| Valores de tratamiento no permitidos, o un solo nivel de tratamiento | `BLOCK` | `validation.treatment` |
| Version o hash incompatibles entre contrato, modelo, DAG y artefacto | `BLOCK` | `versions` + `model_metadata.json` |
| Contrato y DAG incompatibles (backdoor no satisfecho) | `BLOCK` | `causal_checks.py` |
| Propensity fuera de `[0.01, 0.99]` | `BLOCK` | `thresholds.positivity.hard_*` |
| Propensity fuera de `[0.05, 0.95]` | `WARN` | contrato `positivity` |
| Nulos en covariables >= 5% / >= 20% | `WARN` / `BLOCK` | `thresholds.missingness` |
| Filas fuera de `[q01, q99]` de referencia >= 10% / >= 30% (por variable) | `WARN` / `BLOCK` | `thresholds.domain` |
| PSI >= 0.10 / >= 0.25 | `WARN` / `WARN` (nivel critical) | `thresholds.drift`; el contrato fija `drift.action: warn` |
| SUTVA | `REVIEW` (`NOT PROVABLE`) | siempre informativo |
| Sensibilidad | `ROBUST` / `REVIEW` | `thresholds.sensitivity` |

Si la validacion produce `BLOCK`, `app.py` llama a `st.stop()` **antes** de estimar. El orden es
`INPUT -> VALIDATE -> DECIDE -> ESTIMATE -> REPORT`, nunca `INPUT -> ESTIMATE -> "parece que hubo drift"`.

## Los umbrales son criterios del proyecto

**No existe un umbral universal para "validez causal".** Estos valores son decisiones de este proyecto,
revisables por quien lo adapte:

- **Propensity 0.05 / 0.01.** 0.05 es una convencion frecuente para recortar propensities extremos; no es
  una ley estadistica. 0.01 marca un solapamiento casi inexistente. Con la poblacion de referencia el
  propensity esta en [0.11, 0.72], muy lejos de ambos.
- **PSI 0.10 / 0.25.** Reglas practicas de la industria (riesgo de credito), no resultados teoricos. Con
  deciles y n = 1200, el ruido de muestreo del PSI ronda 0.01.
- **Nulos 5% / 20%, dominio 10% / 30%.** Criterios de gestion de calidad de datos elegidos para este caso.
  La fraccion fuera de `[q01, q99]` en la propia referencia es ~2% por construccion.
- **Robustez 0.05.** Un confusor no observado necesitaria explicar al menos 5% de la varianza residual de
  `T` y de `Y` para que el IC incluya 0. Es una exigencia minima arbitraria; el valor obtenido (22.7%) la supera ampliamente.
- **Refutacion p > 0.05 y tolerancia sintetica 0.40.** Convenciones; con muchas simulaciones cada refutador
  puede fallar por azar con probabilidad ~5%.

Cualquier cambio en `thresholds.yaml` debe justificarse aqui y volver a pasar `scripts/validate_release.py`.
