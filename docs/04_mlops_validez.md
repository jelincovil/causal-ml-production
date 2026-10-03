# 04. MLOps y validez

## Ciclo del release

```bash
python scripts/train_model.py                 # artifacts/model/{causal_model.joblib, model_metadata.json}
python scripts/generate_reference_profile.py  # artifacts/monitoring/reference_profile.json
python scripts/validate_release.py            # artifacts/validation/*.json + artifacts/model/model_card.json
```

`validate_release.py` sale con codigo **1** (y no hay release) si:

1. el hash de `data/dataset.csv` no coincide con el del modelo o el perfil de referencia;
2. el contrato y `thresholds.yaml` son incoherentes (banda de positividad);
3. el estimando no es identificable segun DoWhy;
4. la poblacion de referencia incumple su propio contrato (cualquier `BLOCK`);
5. falla una refutacion marcada `required` (placebo, causa comun aleatoria, subconjunto de datos);
6. falta el analisis de sensibilidad requerido;
7. la prueba sintetica no recupera el efecto conocido dentro de la tolerancia;
8. `model_card.json` incumple `contracts/causal-model-card-v1.schema.json`.

`model_card.json` incluye la seccion `supuestos` (catalogo de `validation/assumptions.py`) que la app muestra en pantalla;
los tests fallan si la card commiteada difiere del catalogo, asi que tras cambiar un supuesto hay que volver a ejecutar el release.

```text
validate_release.py -> EXIT 0 -> merge/deploy        EXIT 1 -> NO RELEASE
```

## CI

`.github/workflows/ci.yml` ejecuta `pytest` y `validate_release.py` en cada pull request y push a `main`
(Python 3.12). Codigo que rompe una condicion obligatoria produce un fallo de CI. Para que sea una barrera
real, active la proteccion de rama en GitHub y exija el check `validity-gate`.

## Despliegue

Streamlit Community Cloud toma la app del repositorio de GitHub; un push se refleja en el despliegue y un
cambio de dependencias provoca un redeploy completo. Por eso:

- `requirements.txt` **fija versiones exactas**. `causal_model.joblib` es un pickle de econml/scikit-learn:
  si se actualizan, hay que reentrenar y revalidar. `validation/consistency_checks.py` emite `WARN` si las
  versiones instaladas difieren (major.minor) de las del entrenamiento, y `BLOCK` si el hash del artefacto no
  coincide con `model_metadata.json`.
- Python 3.12 (valor por defecto de Community Cloud; se puede fijar en *Advanced settings* al crear la app).
- `.streamlit/config.toml` queda en la raiz y solo define el tema; no se desactiva CORS ni XSRF.
- No hay `packages.txt`: ninguna dependencia de este repositorio necesita paquetes del sistema. Agreguelo
  solo si una instalacion concreta lo exige y justifique cada paquete.
- `st.cache_resource` (en `app.py`) solo evita recargar contrato, modelo y perfil. El cache optimiza la
  operacion; la validez la protege el contrato.

## Secrets

Cree `.streamlit/secrets.toml` en local (esta en `.gitignore`) o pegue el contenido en *App settings > Secrets*:

```toml
[history]
url = "postgresql+psycopg2://usuario:clave@host:5432/base"
```
