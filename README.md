# Score de originación PD: despliegue

Puesta en producción de los dos modelos supervisados ganadores (XGBoost campeón y CatBoost retador, escenario E3, 9 variables en WoE, calibración sigmoide) y de los tres detectores no supervisados (Isolation Forest, GMM y ECOD) como contraste. No se reentrena nada.

## Estructura

```
DESPLIEGUE/
├── config/main.yaml          parámetros: topes, cut-off, margen, LGD, cortes p99, nº de críticos
├── src/
│   ├── cadena_pd.py          TratadorCrudo y CompactaNoSup: crudo -> tratado (aprendido en entrenamiento)
│   ├── puntuar.py            crudo -> 9 WoE -> 2 PD -> decisión -> 3 categorías no supervisadas
│   ├── simular.py            generador de solicitudes nuevas aleatorias
│   └── ingenieria_pd.py, modelado_pd.py, no_supervisados_pd.py, preprocesamiento_pd.py   clases que necesitan los .pkl
├── models/                   cadenas, modelos, calibradores, tabla WoE, referencias y base_clientes/ (1,347,681 clientes ya puntuados)
├── scripts/construir_base.py regenera la base y las referencias desde SCORE_ORIGINACION
├── simular_solicitudes.py    crea un CSV de solicitudes simuladas
├── data/ejemplos/            CSV de ejemplo para la pestaña 3
├── app/app.py                tablero Streamlit
├── verificar.py              control de que la cadena reproduce los cuadernos
└── requirements.txt          mismas versiones con que se guardaron los .pkl
```

## Pestañas del tablero

| Pestaña | Qué hace |
|---|---|
| Consultar un cliente | Se digita un N.º de cliente (1 a 1,347,681; en una entidad real, la cédula). Muestra PD de XGBoost y CatBoost frente al cut-off, "Crédito aprobado / no aprobado", gráficos del cliente (termómetro, posición en la cartera, SHAP de cada modelo, perfil por percentiles, default histórico de su tramo, tarjeta de puntos) y, abajo, el contraste de los tres detectores. |
| Vigilancia: los 30 más críticos | Sin consulta previa: las 30 solicitudes de toda la base que aún se aprobarían pero están más cerca del cut-off, con ambas PD (seis decimales), minibarra de cercanía y las 3 categorías no supervisadas. Empates: PD CatBoost, luego puntaje IF, luego N.º de cliente; la columna "Empata con" dice cuántos comparten la PD exacta. |
| Ingesta de nuevos clientes | Sube un CSV con valores crudos o genera solicitudes simuladas. Cada fila pasa por las mismas funciones ajustadas en entrenamiento (tramos WoE para los supervisados, MinMax para los detectores). Misma tabla que la pestaña 2. |
| Política y sensibilidad | Efecto del cut-off, margen y LGD sobre la validación. |
| Sobre el modelo | Ficha del modelo y métricas. |

La barra lateral elige el cut-off: umbral F1 (18.5 %) o corte de costo margen/(margen+LGD). Es el mismo para ambos modelos.

## Cómo probarlo en tu PC

```bash
cd "C:\Users\prestige\Documents\UTPL\TUTORIAS TITULACION\DESARROLLO\SCORE_ORIGINACION\DESPLIEGUE"
.venv\Scripts\activate
pip install -r requirements.txt        # agrega catboost
python verificar.py                    # debe terminar con RESULTADO: OK
streamlit run app/app.py
```

## Regenerar la base (solo si cambian los modelos)

```bash
python scripts/construir_base.py "C:\Users\prestige\Documents\UTPL\TUTORIAS TITULACION\DESARROLLO\SCORE_ORIGINACION"
```

## Simular solicitudes nuevas

```bash
python simular_solicitudes.py
```

## Hitos

| Hito | Contenido | Estado |
|---|---|---|
| 0 | Inventario, artefactos, cadena de puntuación y control | listo |
| 1 | Tablero Streamlit de 5 pestañas | listo |
| 2 | GitHub (repositorio, `.gitignore`, primer push) | pendiente |
| 3 | Render (`render.yaml`, despliegue y enlace público) | pendiente |
