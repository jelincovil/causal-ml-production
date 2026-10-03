# Datos

`dataset.csv` es la poblacion de entrenamiento y de referencia del modelo: el split `train`
(n = 1200) del escenario `synthetic/inference/ihdp_medium` del *Causal Benchmark Ecosystem*
(licencia MIT, ver `LICENSE`).

| Columna | Rol |
|---|---|
| `x0` ... `x24` | covariables pre-tratamiento (estandarizadas) |
| `treatment` | tratamiento binario (tasa 0.381) |
| `outcome` | resultado continuo |

- **Todos los valores son sinteticos**: los nombres y la estructura se inspiran en IHDP (Hill, 2011),
  pero los valores los genera un modelo estructural documentado. No representan personas reales.
- El simulador conoce el efecto verdadero (`ate_true` = 3.97); la app no lo usa: solo aparece en
  `docs/model_card.md` como referencia de evaluacion.
- El hash SHA-256 de este archivo queda registrado en `artifacts/model/model_metadata.json` y en
  `artifacts/monitoring/reference_profile.json`; `scripts/validate_release.py` falla si cambian.
  Si reemplaza el dataset debe reentrenar y regenerar el perfil (ver README).
