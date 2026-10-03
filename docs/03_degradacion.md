# 03. Degradacion y monitoring

La validez del efecto estimado depende de que la poblacion de entrada se parezca a la de entrenamiento.
La comparacion no se hace "contra la nada": `artifacts/monitoring/reference_profile.json` guarda, por
covariable, media, desviacion, minimo, maximo, `q01`, `q99`, bordes y proporciones de deciles (para PSI),
una malla de 101 cuantiles (para KS aproximado), la tasa de nulos y la tasa de tratamiento.

```text
poblacion de referencia -> reference_profile.json -> nueva poblacion -> monitoring/drift.py -> comparacion
```

## Metricas

- **PSI por variable** con los deciles de referencia. Es la metrica que activa `WARN`.
- **KS aproximado**: estadistico de una muestra contra la CDF de referencia reconstruida con la malla de
  cuantiles. Es informativo (la referencia no guarda la muestra completa).
- **Tasa de tratamiento** contra la de referencia (informativa: un cambio de asignacion es una senal de
  cambio del mecanismo de seleccion aunque las covariables no se muevan).
- **Fraccion fuera de dominio** y **tasa de nulos**.

## Escenarios de demostracion (sidebar de la app)

| Escenario | Resultado esperado |
|---|---|
| Referencia | `PASS`, estimacion permitida |
| Drift moderado (x0-x2 +0.4 sd) | `WARN` (PSI ~0.17) |
| Drift severo (x0-x2 +1.0 sd) | `WARN` nivel critical (PSI ~0.97): el contrato solo advierte |
| Fuera de dominio (x0-x2 x3) | `BLOCK` por dominio operacional |
| Solapamiento roto (tratamiento = f(x0)) | `BLOCK` por positividad |
| Nulos (10% en x5) | `WARN`; se estima con casos completos |
| Esquema roto (columna `row_id`) | `BLOCK` por schema |
| Tratamiento invalido (valor 2) | `BLOCK` por codificacion del tratamiento |

## Historial y persistencia

Streamlit Community Cloud **no garantiza persistencia del disco local**: un `monitoring_history.csv`
escrito por la app puede desaparecer entre sesiones, asi que no se usa. El historial funciona asi:

- **Sin configuracion:** vive en la sesion (`st.session_state`). La app lo indica en pantalla.
- **Con `[history].url` en `st.secrets`:** se escribe en una base SQL externa (`monitoring/history.py`,
  tabla `monitoring_history`). Para PostgreSQL agregue el driver (p. ej. `psycopg2-binary`) a `requirements.txt`.

Solo con base externa es cierto decir "monitorizamos la degradacion en produccion"; sin ella la app
detecta cambios respecto a la referencia en cada evaluacion, pero no conserva la serie temporal.
