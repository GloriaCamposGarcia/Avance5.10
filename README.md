# Pipeline OSINT & AML Risk Baseline

Este proyecto implementa una arquitectura funcional, modular, trazable y reproducible para la resolución de variantes de nombres de entidades (físicas y morales) mediante técnicas semánticas y el entrenamiento de modelos supervisados de priorización de riesgo AML/fraude a partir de datos OSINT.

## 1. Objetivo del Proyecto
1. **Resolución de Entidades (Pre-OSINT)**: Normalizar nombres de entidades y generar variantes controladas para expandir la cobertura de consultas de fuentes abiertas (OSINT) sin introducir ruido. Utilizar blocking para reducir la complejidad computacional y embeddings (TF-IDF/OpenAI) con similitud coseno para clasificar la coincidencia en niveles de confianza.
2. **Priorización de Riesgo AML (Post-OSINT)**: Entrenar un modelo clasificador baseline (supervisado) que aprenda a priorizar alertas y estimar la probabilidad de riesgo de fraude/AML a partir de las variables estructuradas obtenidas de la ejecución de fuentes OSINT.

---

## 2. Estado Actual
El proyecto ha sido refactorizado y separado en:
- Una biblioteca interna reutilizable en `src/`.
- Scripts de automatización por lotes en `scripts/` para ejecutar ambas fases del pipeline de manera independiente.
- Pruebas unitarias de calidad en `tests/`.
- Conclusiones y análisis teóricos en `docs/`.

Los datasets de entrada se leen desde `data/raw/` y los resultados se escriben en `data/processed/`.

---

## 3. Estructura de Archivos y Carpetas

A continuación se detalla la jerarquía y el propósito de cada elemento del repositorio:

```
Avance5.10/
├── data/
│   ├── raw/                  # Colocar los datasets CSV fuente aquí.
│   │   ├── entity_source_results.csv  # Resultados de consultas y metadatos de entidades.
│   │   └── evidence_items.csv         # Evidencias recuperadas para las entidades.
│   └── processed/            # Ubicación de los CSVs, modelos y gráficos generados.
│       ├── similarity_results.csv          # Similitudes coseno calculadas.
│       ├── manual_review_queue.csv         # Cola de revisión manual priorizada.
│       ├── best_model.pkl                  # Pipeline del mejor clasificador (Random Forest).
│       ├── metrics_comparison.csv          # Tabla comparativa de todos los modelos y tiempos.
│       ├── run_manifest.json               # Metadatos de la corrida.
│       ├── test_metrics.csv                # Métricas de evaluación final en test.
│       ├── test_predictions.csv            # Predicciones de test (reales, predichas, scores).
│       ├── osint_risk_scores.csv           # Scores operativos continuos calculados.
│       ├── confusion_matrix.png            # Matriz de confusión del modelo ganador.
│       ├── roc_curve.png                   # Curva ROC del modelo ganador.
│       ├── precision_recall_curve.png      # Curva Precision-Recall del modelo ganador.
│       └── feature_importance.png          # Importancia de variables por permutación.
├── docs/
│   ├── Embeddings_Resolution_Conclusions.md # Conclusiones de normalización y embeddings.
│   ├── AML_Baseline_Risk_Model_Conclusions.md # Análisis de modelos (ensambles), importancia y score.
│   └── Comparative_Analysis_Sintetico_vs_Real.md # Comparativo entre corridas de datos sintéticos y reales.
├── src/                      # Biblioteca de lógica de negocio del proyecto.
│   ├── __init__.py
│   ├── normalization.py      # Normalización y limpieza de nombres.
│   ├── variants.py           # Algoritmo de generación de 7 reglas de variantes.
│   ├── blocking.py           # Algoritmo de indexación y blocking.
│   ├── embeddings.py         # Motores vectoriales (TF-IDF, OpenAI, Sentence-Transformers) y similitud.
│   └── models.py             # Preprocesamiento, estimadores, stacking y prioridad.
├── scripts/                  # Scripts ejecutables de automatización.
│   ├── __init__.py
│   ├── 01_run_embeddings.py  # Ejecución de la Fase 1 (Embeddings).
│   └── 02_run_aml_baseline.py # Ejecución de la Fase 2 (Entrenamiento AML).
├── tests/                    # Suites de pruebas unitarias.
│   ├── __init__.py
│   ├── test_normalization.py # Validaciones sobre limpieza de nombres.
│   └── test_variants.py      # Validaciones sobre generación de variantes.
├── .gitignore                # Exclusión de archivos locales de Git.
└── README.md                 # Este documento de especificación técnica.
```

---

## 4. Normas del Entorno

Para ejecutar los pipelines con soporte de modelos de lenguaje o incrustaciones semánticas, es necesario configurar variables de entorno locales.

### Gestión Segura de Credenciales (`.env`)
El proyecto incluye soporte integrado para leer variables de entorno locales desde un archivo `.env` ubicado en la raíz del proyecto. El archivo `.env` está registrado de forma explícita en `.gitignore`, por lo que **nunca se subirá al repositorio Git remoto**, garantizando la confidencialidad de tus credenciales.

#### Pasos para Configurar tu Entorno Local:
1. **Instalar Dependencia**: Asegúrate de tener instalada la biblioteca `python-dotenv`:
   ```bash
   pip install python-dotenv
   ```
2. **Crear el archivo `.env`**: En la raíz de este proyecto (al mismo nivel que `README.md` y `.gitignore`), crea un archivo de texto llamado exactamente `.env`.
3. **Definir Variables**: Abre el archivo e ingresa las siguientes variables según sea tu caso:
   ```ini
   # --- CONFIGURACIÓN DE CONEXIÓN CON OPENAI ---
   # Ingresa tu clave secreta de OpenAI sin comillas ni espacios extras.
   # Ejemplo: OPENAI_API_KEY=sk-proj-XXXXX...
   OPENAI_API_KEY=tu_clave_secreta_aqui

   # --- CONFIGURACIÓN DE BACKEND DE EMBEDDINGS ---
   # Valores posibles: 
   # - 'tfidf' (local/gratuito, por defecto)
   # - 'openai' (remoto/pago, requiere API KEY)
   # - 'sentence-transformers' (local/semántico, requiere instalar dependencias)
   EMBEDDING_BACKEND=tfidf

   # --- CONFIGURACIÓN DE MODELOS DE EMBEDDINGS ---
   # Modelo de embedding a utilizar si backend es 'openai'
   OPENAI_EMBEDDING_MODEL=text-embedding-3-small

   # Modelo local de Hugging Face a utilizar para Sentence-Transformers
   HF_EMBEDDING_MODEL=all-MiniLM-L6-v2
   ```

#### Consideraciones Críticas de Seguridad y Uso:
* **Nunca expongas tu clave**: Bajo ninguna circunstancia escribas tu `OPENAI_API_KEY` directamente en el código de `src/` o `scripts/`. Utiliza siempre `os.getenv("OPENAI_API_KEY")`.
* **Alternativas de Embeddings sin Costo**:
  * **TF-IDF**: Por defecto, ejecuta localmente por n-gramas de caracteres sin necesidad de instalar dependencias pesadas.
  * **Sentence-Transformers**: Ejecuta embeddings vectoriales semánticos locales de Hugging Face, de forma gratuita y sin internet, pero requiere instalar las dependencias asociadas (`sentence-transformers` y `torch`).
* **Declaración sin comillas**: En los archivos de entorno `.env` en Python, no es necesario encerrar las cadenas de texto o llaves API entre comillas (por ejemplo, usa `CLAVE=mi_valor` en lugar de `CLAVE="mi_valor"`).

---

## 5. Requisitos Operativos
El proyecto requiere las siguientes bibliotecas de Python para funcionar de manera óptima:
- `pandas`
- `numpy`
- `scipy`
- `scikit-learn`
- `joblib`
- `python-dotenv` (opcional, para cargar archivos `.env`)
- `sentence-transformers` (opcional, requerido para habilitar el backend local de Hugging Face)
- `torch` (opcional, dependencia implícita de sentence-transformers)

---

## 6. Reglas Clave
1. **Calidad de Datos**: Las variantes vacías o nulas se filtran automáticamente antes de indexar.
2. **Control de Target Leakage**: Al entrenar modelos, el script de modelado excluye automáticamente identificadores (`entity_id`), textos descriptivos (`finding_summary`, `context_summary`), campos JSON crudos, y variables de decisión directa (`overall_decision`).
3. **Estratificación**: La partición de datos se hace de forma estratificada 60% entrenamiento, 20% validación y 20% prueba para garantizar la representatividad de la clase minoritaria.

---

## 7. Comandos para la Ejecución

### Ejecución de Pruebas Unitarias
Para verificar la calidad del código y el correcto comportamiento de la normalización y variantes:
```bash
python -m unittest discover -s tests
```

### Fase 1: Embeddings y Similitud
Para ejecutar el pipeline de normalización, variantes, blocking, cálculo de similitudes coseno y generación de cola de revisión priorizada:
```bash
python scripts/01_run_embeddings.py
```
*Salidas generadas en `data/processed/`: `similarity_results.csv` y `manual_review_queue.csv`.*

### Fase 2: Entrenamiento de Modelado AML
Para ejecutar el preprocesamiento, partición, optimización de hiperparámetros de modelos baseline y ensambles homogéneos/heterogéneos (Random Forest, Gradient Boosting, Stacking Classifier), medición de tiempos, exportación de la tabla comparativa de modelos y generación automatizada de curvas e importancias de diagnóstico:
```bash
python scripts/02_run_aml_baseline.py
```
*Salidas generadas en `data/processed/`: `best_model.pkl`, `metrics_comparison.csv`, `run_manifest.json`, `test_metrics.csv`, `test_predictions.csv`, `osint_risk_scores.csv` y los 4 archivos de gráficos de diagnóstico (`confusion_matrix.png`, `roc_curve.png`, `precision_recall_curve.png`, `feature_importance.png`).*
