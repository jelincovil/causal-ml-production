# causal-ml-production

**ML causal en la nube con validez operacional.** App de Streamlit que estima un efecto causal (ATE) **solo si
los datos de entrada cumplen un contrato verificable**: si lo incumplen, la estimacion se bloquea; si hay
condiciones dudosas, se estima con advertencia; si todo pasa, se estima. El repositorio entero (modelo,
contrato, evidencia, referencia de monitoring y CI) viaja junto con la app, de modo que **cada estimacion es
trazable a la version que la produjo**.

> Material docente con datos **sinteticos** (benchmark IHDP-style). No usar para decisiones clinicas ni de
> politica publica reales. La app verifica condiciones operacionales; **no demuestra** ignorabilidad ni
> ausencia de interferencia (ver [Que se verifica y que no](#que-se-verifica-y-que-no)).

Resultado de referencia: **ATE = 3.527** (SE 0.326, IC 95% [2.888, 4.166], n = 1200), estimado con
double machine learning (PLR) y validado con refutaciones de DoWhy.

## Contenido

1. [La idea](#la-idea)
2. [Como se implementa en la nube](#como-se-implementa-en-la-nube)
3. [Modelo y algoritmos](#modelo-y-algoritmos)
4. [El contrato causal](#el-contrato-causal)
5. [Gates: que se comprueba en cada ejecucion](#gates-que-se-comprueba-en-cada-ejecucion)
6. [Evidencia del release](#evidencia-del-release)
7. [Monitoring e historial](#monitoring-e-historial)
8. [La aplicacion](#la-aplicacion)
9. [Estructura del repositorio](#estructura-del-repositorio)
10. [Ejecutar, probar y publicar un release](#ejecutar-probar-y-publicar-un-release)
11. [Desplegar en GitHub y Streamlit Community Cloud](#desplegar-en-github-y-streamlit-community-cloud)
12. [Adaptarlo a su propio modelo y datos](#adaptarlo-a-su-propio-modelo-y-datos)
13. [Que se verifica y que no](#que-se-verifica-y-que-no)
14. [Documentacion, origen y licencia](#documentacion-origen-y-licencia)

---

## La idea

Los supuestos de identificacion (ignorabilidad, positividad, consistencia, SUTVA, DAG correcto) **no
desaparecen al desplegar un modelo causal**. Un estimador entrenado hoy sigue siendo valido solo mientras la
pregunta causal, el DAG, la definicion del tratamiento, los datos de entrada y el soporte estadistico sigan
siendo compatibles con lo que se entreno. Este repositorio convierte esa condicion en un mecanismo ejecutable:

```text
SUPUESTO -> CONTRATO -> CHECK -> GATE -> EVIDENCIA -> MONITORING -> DECISION DE EJECUCION
```

| Etapa | Donde vive | Que hace |
|---|---|---|
| Supuesto | `docs/01_supuestos.md` | enumera que se supone y que se puede comprobar |
| Contrato | `contracts/` | declara estimando, DAG, schema, versiones, reglas y umbrales |
| Check | `validation/`, `monitoring/` | comprueba cada condicion con codigo |
| Gate | `validation/release_gate.py` | traduce checks en `PASS` / `WARN` / `BLOCK` |
| Evidencia | `artifacts/validation/`, `artifacts/model/model_card.json` | resultados versionados del release |
| Monitoring | `artifacts/monitoring/reference_profile.json` | poblacion de referencia para detectar degradacion |
| Decision | `app.py` | estima, estima con advertencia o llama a `st.stop()` |

El orden importa: **primero se valida, despues se estima**.

```text
INPUT -> VALIDATE -> DECIDE -> ESTIMATE -> REPORT          (correcto)
INPUT -> ESTIMATE -> "parece que hubo drift"               (lo que se evita)
```

---

## Como se implementa en la nube

```text
   Desarrollo local                      GitHub                         Streamlit Community Cloud
 ┌──────────────────────┐        ┌───────────────────────┐        ┌──────────────────────────────┐
 │ train_model.py       │        │ repositorio = fuente  │        │ ejecuta app.py desde la raiz │
 │ generate_reference_  │  push  │ de la app y del       │ deploy │ instala requirements.txt     │
 │   profile.py         ├───────►│ argumento cientifico  ├───────►│ (versiones fijadas, Py 3.12) │
 │ validate_release.py  │        │                       │        │ carga contrato, modelo,      │
 │  (DoWhy, EconML)     │        │ CI (ci.yml): pytest + │        │ perfil de referencia y       │
 └──────────────────────┘        │ validate_release.py   │        │ evidencia desde el repo      │
                                 │ ──► check validity-   │        │ por cada sesion:             │
                                 │     gate              │        │ validar -> decidir -> estimar│
                                 └───────────────────────┘        └──────────────┬───────────────┘
                                                                                 │ opcional (st.secrets)
                                                                      ┌──────────▼───────────┐
                                                                      │ base SQL externa:    │
                                                                      │ historial de         │
                                                                      │ monitoring           │
                                                                      └──────────────────────┘
```

**Que ocurre offline (en el release) y que ocurre en la nube (en runtime):**

| | Offline: `scripts/` + CI | Runtime: app en la nube |
|---|---|---|
| Entrenamiento del estimador | `train_model.py` ajusta `LinearDML` y guarda `causal_model.joblib` + metadatos con su SHA-256 | solo carga el artefacto |
| Identificacion y refutaciones | DoWhy (placebo, causa comun aleatoria, subconjunto) con 50 simulaciones, guardadas como JSON | solo **presenta** el JSON; no las recalcula |
| Sensibilidad, prueba sintetica, model card | `validate_release.py` | se muestra la sensibilidad de la referencia y la recalculada sobre la entrada |
| Poblacion de referencia | `generate_reference_profile.py` guarda cuantiles, deciles, tasas | compara la entrada contra ese perfil (PSI, KS, dominio, nulos) |
| Contrato | se valida su coherencia con `thresholds.yaml` y con los hashes de datos y modelo | se aplica a cada entrada: schema, tratamiento, versiones, positividad, dominio, drift |
| Estimacion | prueba sintetica con efecto conocido | ATE de referencia + ATE reajustado sobre la entrada (menos de un segundo con n = 1200) |
| Decision | exit code 0 o 1: hay o no release | `PASS` / `WARN` estima; `BLOCK` ejecuta `st.stop()` antes de estimar |

Decisiones de diseno que la nube hace necesarias:

- **El repositorio es el release.** Streamlit Community Cloud despliega lo que hay en GitHub; por eso modelo,
  contrato, perfil de referencia y evidencia estan versionados en `artifacts/` y `contracts/`.
- **Versiones exactas.** `causal_model.joblib` es un pickle de EconML y scikit-learn: una actualizacion
  automatica puede cambiar el comportamiento o romper la carga. `validation/consistency_checks.py` bloquea si
  el hash del artefacto no coincide con `model_metadata.json` y avisa si cambia la version mayor.menor de una
  libreria respecto del entrenamiento.
- **La nube no garantiza disco.** El disco local de Community Cloud no es persistente, asi que **no se usa un
  CSV local como historial**. Con `[history].url` en `st.secrets` el historial va a una base SQL externa; sin
  eso vive solo en la sesion y la app lo indica en pantalla.
- **La cache no es la validacion.** `st.cache_resource` solo evita recargar contrato, modelo y perfil; la
  validez la protege el contrato, que se evalua en cada ejecucion.
- **Sin configuracion peligrosa.** `.streamlit/config.toml` solo define el tema (CORS y XSRF quedan en sus
  valores por defecto) y no hay `packages.txt` porque ninguna dependencia necesita paquetes del sistema.
- **Falla rapido y no sobrescribe.** Si los datos o artefactos son inconsistentes, `validate_release.py` sale con
  exit 1 en segundos, sin ejecutar DoWhy ni reemplazar la evidencia anterior.

---

## Modelo y algoritmos

**Estimador (`causal_model/estimator.py`):** double machine learning con regresion parcialmente lineal (PLR),
`econml.dml.LinearDML`, `X = None` y las 25 covariables como controles.

- Nuisances: `LassoCV` para `E[Y|X]` y para `E[T|X]` (tratamiento binario residualizado como continuo).
- Cross-fitting con 5 folds; la etapa final usa todos los residuales fuera de muestra (variante DML2).
- Estimando: **ATE** = E[Y(1) - Y(0)]. Bajo PLR el efecto es homogeneo: no hay efecto individual (CATE).
- La app devuelve dos estimaciones que no deben confundirse: la **de referencia** (modelo desplegado) y la
  **reajustada sobre la entrada** (mismo estimador sobre sus casos completos, solo si no hubo `BLOCK`).

**Algoritmos de validacion:**

| Algoritmo | Modulo | Uso |
|---|---|---|
| Criterio de puerta trasera (d-separacion, NetworkX) | `causal_model/graph.py` | el ajuste por las covariables identifica el ATE y ninguna es descendiente de T |
| Identificacion backdoor (DoWhy 0.14) | `causal_model/identification.py` | el estimando es identificable a partir del DAG (release) |
| Propensity: regresion logistica con cross-fitting | `causal_model/propensity.py` | positividad: minimo, maximo, observaciones extremas, ESS/n |
| Refutadores de DoWhy | `causal_model/refutation.py` | `placebo_treatment`, `random_common_cause`, `data_subset` |
| Valor de robustez (Cinelli y Hazlett, 2020) | `validation/sensitivity_checks.py` | fuerza que necesitaria un confusor no observado para anular el efecto |
| PSI por deciles y KS aproximado | `monitoring/drift.py` | distribution shift contra la poblacion de referencia |
| ICC y saturacion por cluster | `validation/sutva_checks.py` | senales de interferencia, si hay variable de agrupacion |
| Simulacion con efecto conocido | `validation/synthetic.py` | el estimador recupera un ATE verdadero (2.0) bajo confusion observada |

Detalle y limites en [`docs/methodology.md`](docs/methodology.md).

**Datos:** `data/dataset.csv` es el split train (n = 1200) de `synthetic/inference/ihdp_medium` del Causal
Benchmark Ecosystem: 25 covariables pre-tratamiento (`x0`...`x24`), `treatment` binario (tasa 0.381) y `outcome`
continuo. Todos los valores son sinteticos. Ver [`data/README.md`](data/README.md).

---

## El contrato causal

Tres archivos en `contracts/` hacen que las reglas **dejen de vivir en la cabeza del investigador**:

| Archivo | Declara |
|---|---|
| `causal_contract.yaml` | modelo y versiones (tratamiento, preprocesamiento, DAG), tratamiento, outcome, estimando, covariables pre-tratamiento, especificacion del estimador y la **accion** (`block` / `warn` / `required`) de cada regla |
| `feature_schema.json` | columnas esperadas: tipo, rol, obligatoriedad, nulabilidad, valores permitidos |
| `thresholds.yaml` | umbrales numericos de PASS / WARN / BLOCK |

Umbrales vigentes (`thresholds.yaml`):

| Condicion | WARN | BLOCK |
|---|---|---|
| Propensity fuera de la banda | `< 0.05` o `> 0.95` | `< 0.01` o `> 0.99` |
| Nulos en una covariable | `>= 5%` | `>= 20%` |
| Filas fuera de `[q01, q99]` de referencia (por variable) | `>= 10%` | `>= 30%` |
| PSI de una covariable | `>= 0.10` | no bloquea: el contrato fija `drift.action: warn` |

> **No existe un umbral universal de "validez causal".** Son criterios operacionales de este proyecto, justificados
> en [`docs/02_traduccion_operacional.md`](docs/02_traduccion_operacional.md); el 0.05 de propensity no es una
> ley estadistica. Quien adapte el proyecto debe revisarlos.

---

## Gates: que se comprueba en cada ejecucion

`validate_runtime_data()` (`validation/release_gate.py`) evalua 11 filas y las muestra siempre en el panel
**Causal validity status**, antes de cualquier estimacion:

| Verificacion | Que comprueba | Puede dar |
|---|---|---|
| Schema | columnas faltantes o desconocidas, tipos no numericos, nulos en `treatment`/`outcome` | PASS, BLOCK |
| Treatment coding | valores permitidos `{0, 1}` y presencia de ambos niveles | PASS, BLOCK |
| Versions & integrity | definicion del tratamiento, pipeline, DAG y modelo compatibles; hash del artefacto; versiones de librerias | PASS, WARN, BLOCK |
| Graph / contract | contrato, schema y DAG coherentes; criterio de puerta trasera; estimando soportado | PASS, BLOCK |
| Missingness | tasa de nulos en covariables | PASS, WARN, BLOCK |
| Operational domain | fraccion de filas fuera del rango de referencia | PASS, WARN, BLOCK |
| Positivity | propensity con cross-fitting: minimo, maximo, observaciones extremas, ESS | PASS, WARN, BLOCK |
| Distribution shift | PSI por covariable, KS aproximado, tasa de tratamiento | PASS, WARN |
| SUTVA diagnostics | senales de interferencia si hay variable de agrupacion; si no, `NOT PROVABLE` | siempre REVIEW |
| Refutation (release) | refutaciones guardadas en el release | PASS, WARN |
| Unobserved confounding | valor de robustez de la referencia | ROBUST, REVIEW |

El estado global es el peor entre PASS, WARN y BLOCK; REVIEW, ROBUST y SKIPPED son informativos. Si el esquema o
el tratamiento son invalidos, las verificaciones que dependen de los datos se marcan `SKIPPED`.

```text
BLOCK  -> EFFECT ESTIMATION: BLOCKED   (st.stop(), no se calcula nada)
WARN   -> EFFECT ESTIMATION: ALLOWED con advertencias
PASS   -> EFFECT ESTIMATION: ALLOWED
```

---

## Evidencia del release

`python scripts/validate_release.py` genera la evidencia y **sale con exit 1 (no hay release)** si falla alguna
condicion obligatoria:

| Condicion | Artefacto que produce |
|---|---|
| hashes de dataset, modelo y perfil de referencia coherentes; banda de positividad del contrato igual a `thresholds.yaml` | |
| la poblacion de referencia cumple su propio contrato (sin `BLOCK`) | `artifacts/validation/validation_report.json` |
| el estimando es identificable segun DoWhy | `validation_report.json` |
| refutaciones `required` superadas (placebo, causa comun aleatoria, subconjunto) | `artifacts/validation/refutation_report.json` |
| analisis de sensibilidad presente | `validation_report.json` |
| prueba sintetica: error absoluto <= 0.40 respecto del efecto conocido | `validation_report.json` |
| model card valida contra `contracts/causal-model-card-v1.schema.json` | `artifacts/model/model_card.json` |

Resultados actuales: refutaciones PASS (p = 0.48 / 0.35 / 0.29), valor de robustez 27.0% (22.7% para que el IC
incluya 0), prueba sintetica con error 0.03, propensity de referencia en [0.11, 0.72].

---

## Monitoring e historial

La comparacion no se hace "contra la nada": `artifacts/monitoring/reference_profile.json` guarda por covariable
media, desviacion, minimo, maximo, `q01`, `q99`, bordes y proporciones de deciles (PSI), una malla de 101 cuantiles
(KS aproximado) y tasa de nulos, ademas de la tasa de tratamiento y el hash de los datos de entrenamiento.

```text
poblacion de referencia -> reference_profile.json -> nueva poblacion -> monitoring/drift.py -> comparacion
```

El boton **Guardar esta evaluacion en el historial** registra fecha, version del modelo, estado, n, PSI maximo, KS
maximo, tasa de tratamiento y ATE reajustado. Para que la afirmacion "monitorizamos la degradacion en produccion" sea
cierta hace falta persistencia externa:

```toml
# .streamlit/secrets.toml (local, ignorado por git) o App settings > Secrets en la nube
[history]
url = "postgresql+psycopg2://usuario:clave@host:5432/base"
```

Con PostgreSQL agregue el driver (por ejemplo `psycopg2-binary`) a `requirements.txt`. Sin `[history]` la app usa el
historial de sesion y lo avisa.

---

## La aplicacion

La barra lateral permite elegir un **escenario de demostracion** o subir un CSV propio (columnas de
`contracts/feature_schema.json`). El flujo de `app.py`:

1. Estado de validez causal (11 verificaciones y la decision `ALLOWED` / `BLOCKED`).
2. **Supuestos del analisis**: seccion unica con 16 supuestos en tres niveles (ver abajo). Se muestra siempre, tambien
   cuando hay `BLOCK`.
3. Si hay `BLOCK`: mensaje y `st.stop()`. Si hay `WARN`: aviso y continua.
4. DAG del contrato, panel de positividad (histograma de propensity por grupo, ESS), panel de monitoring (PSI por
   variable, historial), efecto estimado (referencia vs entrada, con IC y sensibilidad) y refutaciones del release.

Escenarios de demostracion y desenlace esperado:

| Escenario | Resultado |
|---|---|
| Referencia (poblacion de entrenamiento) | PASS |
| Drift moderado (x0-x2 +0.4 sd) | WARN (PSI ~0.17) |
| Drift severo (x0-x2 +1.0 sd) | WARN nivel critical (PSI ~0.97): el contrato solo advierte |
| Fuera de dominio (x0-x2 por 3) | BLOCK por dominio operacional |
| Solapamiento roto (tratamiento determinado por x0) | BLOCK por positividad |
| Nulos (10% en x5) | WARN; se estima con casos completos |
| Esquema roto (columna `row_id`) | BLOCK por schema |
| Tratamiento invalido (5% con valor 2) | BLOCK por codificacion del tratamiento |

---

## Estructura del repositorio

```text
causal-ml-production/
├── app.py                         # flujo INPUT -> VALIDATE -> DECIDE -> ESTIMATE -> REPORT
├── requirements.txt               # versiones exactas (runtime, Python 3.12)
├── requirements-dev.txt           # + pytest
├── pytest.ini
├── LICENSE
├── .streamlit/config.toml         # solo tema
├── .github/workflows/ci.yml       # pytest + validate_release.py (check validity-gate)
├── contracts/
│   ├── causal_contract.yaml       # contrato ejecutable
│   ├── feature_schema.json        # schema de entrada
│   ├── thresholds.yaml            # umbrales PASS/WARN/BLOCK
│   └── causal-model-card-v1.schema.json
├── causal_model/                  # graph, identification, estimator, propensity, refutation
├── validation/                    # checks, release_gate, results, synthetic, model_card, assumptions (catalogo de supuestos)
├── monitoring/                    # reference_profile, runtime_profile, drift, monitoring_report, history
├── components/                    # paneles de Streamlit (incluye assumptions_panel) + data_input (escenarios y CSV)
├── artifacts/
│   ├── model/                     # causal_model.joblib, model_metadata.json, model_card.json
│   ├── validation/                # validation_report.json, refutation_report.json
│   └── monitoring/                # reference_profile.json
├── data/                          # dataset.csv (sintetico) y README
├── docs/                          # 01_supuestos ... 04_mlops_validez, methodology, model_card
├── scripts/                       # train_model, generate_reference_profile, validate_release
└── tests/                         # 83 tests (grafo, estimador, schema, gates, monitoring, release, supuestos, app)
```

---

## Ejecutar, probar y publicar un release

Requiere Python 3.12.

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
streamlit run app.py
```

```bash
pytest -q                                  # 83 tests, incluida la app con streamlit.testing
python scripts/validate_release.py         # evidencia de validez; exit 1 si falta una condicion obligatoria
```

Ciclo de release cuando cambia el dataset, el contrato o el estimador (este orden):

```bash
python scripts/train_model.py                 # artifacts/model/{causal_model.joblib, model_metadata.json}
python scripts/generate_reference_profile.py  # artifacts/monitoring/reference_profile.json
python scripts/validate_release.py            # artifacts/validation/*.json + artifacts/model/model_card.json
```

```text
validate_release.py -> EXIT 0 -> merge / deploy        EXIT 1 -> NO RELEASE
```

El CI (`.github/workflows/ci.yml`) ejecuta lo mismo con Python 3.12 en cada pull request y push a `main`.

---

## Desplegar en GitHub y Streamlit Community Cloud

```bash
cd streamlit_repo_gh        # esta carpeta es la raiz del repositorio
git init -b main
git add .
git commit -m "Primera version"
git remote add origin https://github.com/<usuario>/<repositorio>.git
git push -u origin main
```

1. Entre a [share.streamlit.io](https://share.streamlit.io) con su cuenta de GitHub y elija **Create app**.
2. Seleccione el repositorio, la rama `main` y el archivo principal `app.py`.
3. En *Advanced settings* elija **Python 3.12**.
4. (Opcional) Pegue el contenido de `secrets.toml` en *Secrets* para un historial persistente.
5. Despliegue. La primera carga tarda porque instala `econml` y `dowhy`.

Un push a `main` redespliega la app y un cambio en `requirements.txt` provoca un redeploy completo. Para que el CI sea
una barrera real, active la proteccion de rama en GitHub y exija el check `validity-gate`. Las versiones fijadas
fueron probadas con una instalacion limpia en Python 3.12 (83 tests y `validate_release.py`).

---

## Adaptarlo a su propio modelo y datos

1. Reemplace `data/dataset.csv` por su poblacion de entrenamiento (sin identificadores ni datos personales).
2. Edite `contracts/causal_contract.yaml` (tratamiento, outcome, covariables **pre-tratamiento**, niveles del
   tratamiento, versiones) y `contracts/feature_schema.json`.
3. Revise `contracts/thresholds.yaml`: son criterios de proyecto, no leyes; justifique los cambios en
   `docs/02_traduccion_operacional.md`.
4. Si cambia la forma del DAG, modifique `causal_model/graph.py`; el criterio de puerta trasera se vuelve a comprobar solo.
5. Ejecute el ciclo de release (`train_model` -> `generate_reference_profile` -> `validate_release`) y corrija lo que el
   gate bloquee.
6. Actualice `docs/model_card.md` y los escenarios de `components/data_input.py` si usa otras columnas.

---

## Que se verifica y que no

| Categoria | Elementos | Mecanismo |
|---|---|---|
| **Comprobable operacionalmente** | schema, codificacion del tratamiento, rangos, nulos, overlap, propensity extremo, distribution shift, versiones e integridad del artefacto | checks que permiten, advierten o bloquean |
| **Evaluable con evidencia indirecta** | interferencia (SUTVA), sensibilidad a confusion no observada, estabilidad | diagnosticos y refutaciones; resultado `REVIEW` / `ROBUST`, nunca `PASS` |
| **No demostrable por una app** | ignorabilidad verdadera, ausencia absoluta de interferencia, validez causal universal | exigen justificacion sustantiva |

Por eso la app no muestra un unico "CAUSAL VALIDITY = PASS": muestra operational checks, supuestos documentados,
refutacion, sensibilidad y la lista de supuestos que no se pueden verificar.

### Seccion "Supuestos del analisis"

La pantalla separa lo que se contrasta con los datos de lo que no, **incluidos los supuestos del metodo**. Cada
supuesto aparece una sola vez, con su estado, la funcion que lo evalua y que se comprueba:

| Nivel | Supuestos |
|---|---|
| 1. Verificados empiricamente | esquema de entrada, tratamiento binario con ambos niveles, positividad, dominio de referencia, ausencia de drift, completitud |
| 2. Evidencia indirecta | ignorabilidad condicional (valor de robustez), efecto no espurio y estable (refutaciones) |
| 3. No verificables por la app | DAG, SUTVA, consistencia del tratamiento, identificacion valida en la poblacion de entrada, datos faltantes ignorables y los tres **supuestos del metodo**: homogeneidad del efecto (PLR con `X=None`), nuisances con LassoCV sin diagnostico de convergencia y normalidad asintotica del IC |

- **Una sola fuente.** El texto vive en `validation/assumptions.py`; `validate_release.py` lo escribe en la seccion
  `supuestos` de `artifacts/model/model_card.json` y la app renderiza esa seccion sin texto propio. Los tests fallan si
  la card y el catalogo divergen, si una entrada del contrato queda sin supuesto, o si una declaracion deja de
  coincidir con el codigo (por ejemplo, si el estimador deja de usar `LassoCV` o se agrega un diagnostico de nuisances).
- **Solo informa.** No cambia ningun veredicto `PASS` / `WARN` / `BLOCK` ni los umbrales. Los supuestos del nivel 3 muestran
  `NO VERIFICADO`, nunca `PASS`.
- **Operacional vs indirecto.** El expander "Lectura del estado causal" muestra el estado de las verificaciones
  operacionales (sin SUTVA, refutacion ni sensibilidad) por separado de la evidencia indirecta; el estado global sigue
  combinando ambos grupos.
- **Avisos.** El panel de positividad aclara que el propensity viene de una regresion logistica con cross-fitting, distinta
  del LassoCV del estimador; el efecto re-estimado sobre la entrada advierte que los supuestos de identificacion deben valer en
  esa poblacion; y la fila `Versions & integrity` se declara auditoria de artefactos, no de la consistencia causal.

Limitaciones conocidas: datos sinteticos (no hay validez externa en juego); efecto homogeneo bajo PLR (no apto para
decisiones individuales ni focalizacion); el KS de monitoring es aproximado porque el perfil guarda cuantiles y no la
muestra; el valor de robustez se aplica a los residuales del DML y es una aproximacion; sin base externa el historial
no sobrevive a la sesion. En este escenario el IC contiene el ATE verdadero del simulador (3.97), pero la
estimacion puntual lo subestima en ~0.44.

---

## Documentacion, origen y licencia

| Documento | Contenido |
|---|---|
| [`docs/01_supuestos.md`](docs/01_supuestos.md) | supuestos de identificacion y su estado de verificacion |
| [`docs/02_traduccion_operacional.md`](docs/02_traduccion_operacional.md) | de supuesto a regla ejecutable; justificacion de umbrales |
| [`docs/03_degradacion.md`](docs/03_degradacion.md) | metricas de deriva, escenarios y persistencia |
| [`docs/04_mlops_validez.md`](docs/04_mlops_validez.md) | release, CI, despliegue y secrets |
| [`docs/methodology.md`](docs/methodology.md) | estimador, refutaciones, sensibilidad, limites |
| [`docs/model_card.md`](docs/model_card.md) | model card legible (la machine-readable esta en `artifacts/model/model_card.json`) |
| [`temp/explicacion_modelo.html`](temp/explicacion_modelo.html) | explicacion narrada del modelo, el entrenamiento y la puesta en produccion (HTML autocontenido) |

**Origen.** Resume lo esencial de la Serie B del curso *Sistema de Inteligencia Causal*: el estimador PLR/DML y el
servicio de B1, la model card, el schema y el CI de B2 y el monitoreo de drift de B4, bajo una arquitectura de validez
operacional. La politica de focalizacion de B3 (T-learner con presupuesto del 20%) no se incluye: requiere un
estimador de efectos heterogeneos que este contrato no declara.

**Licencia.** MIT (ver `LICENSE`). Los datos derivan del *Causal Benchmark Ecosystem* (MIT, Segundo Alejandro
Santibanez Uribe).
