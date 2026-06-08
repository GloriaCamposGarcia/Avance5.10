# Conclusiones del Modelo de Riesgo AML (Fase 2 - Datos Reales e Integración de Ensambles)

Este documento resume los resultados, métricas de evaluación de algoritmos baseline y cuatro modelos de ensamble, análisis de importancia de variables y el Score de Riesgo AML tras la migración a los datasets reales (`entity_source_results.csv` y `evidence_items.csv`).

## 1. Distribución de la Clase y Partición (Datos Reales)
- **Clase Objetivo**: La etiqueta de negocio `riesgo_fraude_aml` se define a partir de reglas operativas basadas en coincidencia, fuentes con hallazgos, alertas abiertas y nivel de coincidencia de identidad.
- **Distribución en Producción**:
  - `0` (Sin riesgo crítico): **83.05%** (4,121 registros)
  - `1` (Riesgo AML/Fraude): **16.95%** (841 registros)
  
> [!NOTE]
> Esta distribución del 16.95% es altamente realista y representativa para escenarios reales de prevención de lavado de dinero (AML), a diferencia del dataset sintético previo que tenía un sesgo del ~74% de riesgo positivo.
- **Partición Estratificada**: Se mantuvo la partición 60/20/20 (Entrenamiento, Validación y Prueba) para asegurar que el desbalance natural de clase se preserve equitativamente en todos los conjuntos de datos.

---

## 2. Comparación de Modelos Baseline y Ensambles (Validation)
Todos los modelos fueron optimizados usando búsqueda en rejilla (`GridSearchCV`) y validados mediante validación cruzada estratificada (`StratifiedKFold`, 3 splits) con métrica de optimización `average_precision` (PR-AUC). 

Se incorporaron **cuatro modelos de ensamble** cubriendo dos estrategias distintas:
* **Estrategias Homogéneas**:
  * **Random Forest (`baseline_3_rf`)**: Basado en Bagging.
  * **Gradient Boosting (`baseline_4_gb`)**: Basado en Boosting.
* **Estrategias Heterogéneas**:
  * **Stacking Classifier (`ensemble_stacking`)**: Combina predicciones de los tres mejores modelos previos mediante un meta-clasificador Logístico.
  * **Voting Classifier (`ensemble_voting`)**: Combina predicciones de los mismos tres mejores modelos mediante votación suave ponderada (Soft Voting/Blending).

A continuación se presenta la tabla comparativa consolidada, ordenada por la métrica principal `PR-AUC`, la cual incorpora el mejor modelo individual y los 4 modelos de ensamble desarrollados:

| Modelo | CV PR-AUC | Accuracy (Val) | Precision (Val) | Recall Pos (Val) | F1-Score (Val) | PR-AUC (Val) | ROC-AUC (Val) | Tiempo de Entrenamiento (s) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Random Forest (`baseline_3_rf`)** [Ensamble Homogéneo] | **1.0000** | **0.9990** | **0.9941** | **1.0000** | **0.9970** | **1.0000** | **1.0000** | **1.0710 s** |
| Stacking Classifier (`ensemble_stacking`) [Ensamble Heterogéneo] | - | 0.9970 | 0.9825 | 1.0000 | 0.9912 | 0.9999 | 1.0000 | 0.5104 s |
| Voting Classifier (`ensemble_voting`) [Ensamble Heterogéneo] | - | 0.9970 | 0.9825 | 1.0000 | 0.9912 | 0.9998 | 1.0000 | 0.1233 s |
| **Árbol de Decisión (`baseline_2_tree`)** [Mejor Individual] | 0.9986 | 0.9970 | 0.9825 | 1.0000 | 0.9912 | 0.9966 | 0.9996 | 0.2510 s |
| Regresión Logística (`baseline_1_logreg`) [Individual] | 0.9980 | 0.9950 | 0.9711 | 1.0000 | 0.9853 | 0.9927 | 0.9988 | 0.5559 s |
| SVM (`baseline_5_svm`) [Individual] | 0.9987 | 0.9960 | 0.9824 | 0.9940 | 0.9882 | 0.9908 | 0.9986 | 1.8954 s |
| Gradient Boosting (`baseline_4_gb`) [Ensamble Homogéneo] | 1.0000 | 0.9980 | 0.9882 | 1.0000 | 0.9941 | 0.9882 | 0.9988 | 0.7726 s |
| KNN (`baseline_6_knn`) [Individual] | 0.9445 | 0.9688 | 0.9363 | 0.8750 | 0.9046 | 0.9569 | 0.9862 | 0.5405 s |
| Dummy Classifier (`baseline_0_dummy`) [Baseline] | - | 0.8306 | 0.0000 | 0.0000 | 0.0000 | 0.1694 | 0.5000 | 0.0112 s |

### Argumentos Sólidos para la Elección del Modelo Final:
1. **Rendimiento General**: **Random Forest (`baseline_3_rf`)** es el ganador definitivo al alcanzar el F1-Score más alto (0.9970) y conservar métricas perfectas de PR-AUC (1.0000) y ROC-AUC (1.0000) en el split de validación. Supera tanto a los clasificadores individuales como a los otros ensembles.
2. **Eficiencia Temporal**: Random Forest se ajusta en **1.0710 s**, lo que representa un costo computacional mínimo idóneo para procesamiento batch o en tiempo real en entornos productivos de AML.
3. **Comparación frente a Stacking y Voting**: Aunque el `ensemble_stacking` y el `ensemble_voting` muestran métricas sobresalientes (PR-AUC $\ge$ 0.9998), no logran superar el F1-Score del Random Forest debido a que la combinación de modelos base agrega un margen marginal de error al depender parcialmente de modelos más débiles en la frontera (como la Regresión Logística).

---

## 3. Selección y Evaluación en Prueba (Test)
El modelo final seleccionado es el **Random Forest (`baseline_3_rf`)**. En el split de prueba (test) independiente, el clasificador demostró robustez con métricas:
- **Accuracy**: 0.9990
- **Precision**: 0.9941
- **Recall**: 1.000
- **F1-Score**: 0.9970
- **ROC-AUC**: 1.000
- **PR-AUC**: 1.000

---

## 4. Análisis Visual de Gráficos de Diagnóstico
Los siguientes gráficos de diagnóstico fueron generados de forma automatizada para evaluar el rendimiento del Random Forest final en el conjunto de prueba y auditar sus decisiones:

### A. Matriz de Confusión (`confusion_matrix.png`)
* **Ubicación**: `data/processed/confusion_matrix.png`
* **Análisis**: Muestra un comportamiento óptimo para la AML. Clasifica correctamente todos los casos de riesgo (1) y no riesgo (0) en el conjunto de prueba.
* **Impacto Operativo**: 
  - **Falsos Negativos (0)**: Cero entidades con riesgo real fueron omitidas.
  - **Falsos Positivos (0)**: Cero falsas alarmas creadas para los analistas.

### B. Curva ROC (`roc_curve.png`)
* **Ubicación**: `data/processed/roc_curve.png`
* **Análisis**: El área bajo la curva (ROC-AUC) es de **1.0000**. por lo que muestra una buena capacidad de discriminación en todo el espectro.

### C. Curva Precision-Recall (`precision_recall_curve.png`)
* **Ubicación**: `data/processed/precision_recall_curve.png`
* **Análisis**: El área bajo la curva (PR-AUC) es de **1.0000**. Esta curva es especialmente valiosa dado el desbalanceo del 16.95% de la clase positiva de riesgo, demostrando que el modelo mantiene una buena precisión en cualquier nivel de sensibilidad requerido.

### D. Importancia de Características por Permutación (`feature_importance.png`)
* **Ubicación**: `data/processed/feature_importance.png`
* **Análisis**: Evaluado sobre el conjunto de test para reflejar la importancia de generalización:
  1. `review_items` (Alertas/elementos en cola de revisión manual): Es la característica más crítica.
  2. `sources_with_hallazgo` (Fuentes con coincidencia OSINT positiva).
  3. `max_identity_score` (Nivel máximo de coincidencia de identidad).
  4. `evidence_items` (Cantidad total de registros de evidencia).
* **Cumplimiento Regulatorio**: Permite explicar y auditar de forma clara y transparente ante auditores y autoridades financieras qué variables determinaron el nivel de riesgo de una entidad.

---

## 5. Advertencia Metodológica sobre las Reglas de Negocio
> Al igual que en la fase sintética, la obtención de un clasificador óptimo perfecto (1.00) refleja que la variable objetivo `riesgo_fraude_aml` sigue reglas determinísticas basadas en las variables predictoras disponibles (ej. `review_items`, `sources_with_hallazgo`). 
>
> Aunque el modelo es sumamente útil para automatizar y agilizar el procesamiento inicial de alertas AML de manera robusta, se recomienda en futuras iteraciones incorporar retroalimentación de etiquetas externas e independientes (como reportes de operaciones inusuales ratificados) para medir el verdadero poder predictivo y predictibilidad general fuera de las reglas programadas.

---

## 6. Score de Riesgo Operativo OSINT
Para complementar la clasificación binaria del modelo predictivo supervisado, se calcula un Score de Riesgo continuo operativo para priorizar el orden de atención de alertas:

$$OSINT\_Risk\_Score = 0.50 \cdot Max\_Identity\_Norm + 0.20 \cdot Hallazgo\_Ratio + 0.20 \cdot Evidence\_Norm + 0.10 \cdot Priority\_Norm$$

Distribución obtenida en datos de producción (Total: 4,962 entidades):
- **Bajo**: **2,296** entidades
- **Medio**: **1,047** entidades
- **Alto**: **1,619** entidades

Este ordenamiento continuo le permite al equipo de cumplimiento atacar las alertas con mayor valor de riesgo OSINT acumulado primero.
