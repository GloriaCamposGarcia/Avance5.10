# Conclusiones del Modelo de Riesgo AML (Fase 2)

Este documento resume los resultados, métricas de evaluación, análisis de importancia de variables y la construcción de métricas operacionales de la Fase 2 (Modelo Baseline Supervisado y Score de Riesgo AML).

## 1. Distribución de la Clase y Partición
- **Clase Objetivo**: La etiqueta de negocio `riesgo_fraude_aml` se definió de forma binaria a partir de reglas operativas basadas en la decisión general, las fuentes con hallazgos, el score máximo y las alertas abiertas.
- **Distribución**:
  - `1` (Riesgo AML/Fraude): **74.72%** (7,472 registros)
  - `0` (Sin riesgo crítico): **25.28%** (2,528 registros)
- **Partición Estratificada**: Se dividió el dataset en proporciones 60/20/20. Cada split (entrenamiento, validación y prueba) retuvo exactamente las mismas proporciones de la variable objetivo, evitando sesgos de distribución.

---

## 2. Comparación de Modelos Baseline y Ajuste (Validation)
Se evaluaron y ajustaron por validación cruzada (`StratifiedKFold`, 3 splits) con métrica objetivo `average_precision` los siguientes clasificadores:

| Estimador | Accuracy | Precision | Recall (Clase 1) | F1-Score | PR-AUC | ROC-AUC |
|---|---|---|---|---|---|---|
| **Árbol de Decisión (`baseline_2_tree`)** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **1.0000** |
| Random Forest (`baseline_3_rf`) | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| Gradient Boosting (`baseline_4_gb`) | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| Regresión Logística (`baseline_1_logreg`) | 0.9925 | 1.0000 | 0.9899 | 0.9949 | 0.9999 | 0.9999 |
| SVM (`baseline_5_svm`) | 0.9935 | 0.9933 | 0.9979 | 0.9956 | 0.9999 | 0.9998 |
| KNN (`baseline_6_knn`) | 0.9925 | 0.9920 | 0.9979 | 0.9949 | 0.9995 | 0.9998 |
| Dummy Classifier | 0.7475 | 0.7475 | 1.0000 | 0.8555 | 0.7475 | 0.5000 |

* **Mejor Modelo**: El clasificador **baseline_2_tree** (Árbol de decisión con profundidad máxima ajustada por grid) fue elegido como el modelo final debido a su simplicidad, interpretabilidad y rendimiento perfecto.
* **Evaluación en Prueba (Test)**: El modelo final mantuvo su desempeño perfecto en el test split con métricas de **1.00** en todas las dimensiones y una matriz de confusión sin errores:
  - Falsos Positivos: 0
  - Falsos Negativos: 0

---

## 3. Advertencia Metodológica: Métricas Perfectas
> [!WARNING]
> La obtención de métricas perfectas (1.00) se debe a que la etiqueta objetivo `riesgo_fraude_aml` fue construida mediante reglas lógicas explícitas usando las variables de entrada (`max_identity_score`, `sources_with_hallazgo` y `review_items`).
>
> Por lo tanto, el modelo aprendió la frontera de decisión determinística implícita en la definición del target. Este modelo sirve perfectamente como baseline de automatización, pero requiere validarse con etiquetas independientes de negocio (ej. fraude materializado, reportes de operaciones sospechosas reales) en siguientes iteraciones para medir su verdadero poder predictivo y evitar el sobreajuste de las reglas.

---

## 4. Importancia de Variables
El análisis de importancia por permutación y coeficientes determinó las siguientes variables como las más críticas para separar la clase de riesgo:
1. `max_identity_score`: Máximo score de coincidencia de identidad en fuentes.
2. `sources_with_hallazgo`: Número de fuentes OSINT donde se obtuvo una coincidencia.
3. `review_items`: Elementos listados en colas de revisión manual.
4. `sources_evaluated`: Total de fuentes consultadas.

---

## 5. OSINT Risk Score Operativo
Como complemento al modelo predictivo, se implementó una ecuación de puntaje ponderado continuo (`osint_risk_score`) para priorizar entidades:

$$OSINT\_Risk\_Score = 0.50 \cdot Max\_Identity\_Norm + 0.20 \cdot Hallazgo\_Ratio + 0.20 \cdot Evidence\_Norm + 0.10 \cdot Priority\_Norm$$

Donde:
- `Max_Identity_Norm`: Score máximo de identidad normalizado entre 0 y 1.
- `Hallazgo_Ratio`: Proporción de fuentes con hallazgo respecto a las evaluadas.
- `Evidence_Norm`: Conteo de evidencias encontradas normalizado.
- `Priority_Norm`: Puntuación ponderada de la prioridad de elementos de la cola de revisión.

Este score se distribuye en tres niveles operativos:
- **Bajo**: Score $\le Q_{33}$ (3,352 registros).
- **Medio**: Score entre $Q_{33}$ y $Q_{66}$ (3,256 registros).
- **Alto**: Score $> Q_{66}$ (3,392 registros).
* **Uso**: Permite que el analista ordene de manera continua la cola de alertas prioritarias para enfocar recursos.
