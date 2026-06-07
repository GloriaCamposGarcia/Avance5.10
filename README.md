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
│   │   ├── Entities_Dataset_sintetico.csv  # Datos iniciales de entidades.
│   │   └── entities_osint_homogeneous.csv  # Reportes OSINT estructurados.
│   └── processed/            # Ubicación de los CSVs y modelos generados.
│       ├── similarity_results.csv          # Similitudes coseno calculadas.
│       ├── manual_review_queue.csv         # Cola de revisión manual priorizada.
│       ├── best_model.pkl                  # Pipeline del mejor clasificador (Tree).
│       ├── run_manifest.json               # Metadatos e hiperparámetros de corrida.
│       └── test_metrics.csv                # Métricas de evaluación final.
├── docs/
│   ├── Embeddings_Resolution_Conclusions.md # Conclusiones de normalización y embeddings.
│   └── AML_Baseline_Risk_Model_Conclusions.md # Análisis de modelos, importancia y score.
├── src/                      # Biblioteca de lógica de negocio del proyecto.
│   ├── __init__.py
│   ├── normalization.py      # Normalización y limpieza de nombres.
│   ├── variants.py           # Algoritmo de generación de 7 reglas de variantes.
│   ├── blocking.py           # Algoritmo de indexación y blocking.
│   ├── embeddings.py         # Motores vectoriales (TF-IDF, OpenAI) y similitud.
│   └── models.py             # Preprocesamiento, estimadores y prioridad.
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
- **Configuración del API Key**: Para utilizar el backend semántico de OpenAI, configure la variable de entorno `OPENAI_API_KEY` en su sistema o cree un archivo `.env` en la raíz del proyecto.
- **Selección de Backend**: Puede forzar el uso de un motor vectorial específico seteando la variable de entorno `EMBEDDING_BACKEND` con los valores `tfidf` (por defecto, ganador de la validación) o `openai`.
- **Modelo OpenAI**: Por defecto se utiliza `text-embedding-3-small` para OpenAI, modificable con la variable de entorno `OPENAI_EMBEDDING_MODEL`.

---

## 5. Requisitos Operativos
El proyecto requiere las siguientes bibliotecas de Python para funcionar de manera óptima:
- `pandas`
- `numpy`
- `scipy`
- `scikit-learn`
- `joblib`
- `python-dotenv` (opcional, para cargar archivos `.env`)

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
Para ejecutar el preprocesamiento, partición, entrenamiento con validación cruzada y ajuste de hiperparámetros (GridSearchCV) del clasificador de riesgo:
```bash
python scripts/02_run_aml_baseline.py
```
*Salidas generadas en `data/processed/`: `best_model.pkl`, `run_manifest.json`, `test_metrics.csv` y `test_predictions.csv`.*
