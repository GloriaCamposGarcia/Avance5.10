# Conclusiones de Resolución de Entidades (Fase 1)

Este documento resume las métricas clave, las lecciones metodológicas y los resultados de las pruebas comparativas realizadas durante la Fase 1 del pipeline (Embeddings y Resolución de Variantes de Nombres).

## 1. Normalización de Nombres
La normalización de nombres es la primera línea de defensa contra el ruido nominal. Al procesar las **10,000 entidades originales**, la estandarización (que limpia acentos, caracteres especiales, y remueve sufijos legales como S.A., S. de R.L. de C.V. para entidades morales) redujo la cardinalidad de nombres únicos de **10,000 a 8,768 nombres base únicos**.
* **Impacto**: Esto demuestra que el pipeline remueve efectivamente variaciones ortográficas y corporativas comunes, permitiendo que variantes similares se mapeen a una base canónica común antes de calcular similitudes vectoriales.

---

## 2. Reglas de Generación de Variantes
Se definieron e implementaron 7 reglas de generación para expandir la cobertura de búsquedas en fuentes OSINT:
1. `base_canonical`: Nombre base normalizado.
2. `reverse_order`: Inversión del orden de los tokens (para nombres escritos como Apellido, Nombre).
3. `initials_full`: Iniciales de cada token separadas por espacio.
4. `first_last`: Primer token y último token.
5. `first_initial_last`: Primera inicial y último token.
6. `truncate_3_tokens`: Truncamiento conservador a los primeros 3 tokens.
7. `moral_remove_corp_noise`: (Solo para morales) Remoción de palabras corporativas residuales comunes (ej. group, holdings, trading).

De las 10,000 entidades iniciales, este algoritmo generó un total de **50,674 variantes únicas** (~5 variantes por entidad).

---

## 3. Desempeño del Blocking
El cálculo de similitud *all-vs-all* de las 50,674 variantes implicaría evaluar **1,283,901,801 pares**, lo cual es computacionalmente inviable en producción.
* **Filtros de Blocking**: Se aplicó una estrategia de blocking estricta que empareja variantes únicamente si comparten país, tipo de entidad (Física/Moral) y la misma letra inicial. Adicionalmente, se filtraron pares con una diferencia de longitud de caracteres > 15 y una diferencia de tokens > 2.
* **Métricas de Reducción**:
  - Pares resultantes del blocking: **9,453,802 pares**.
  - **Porcentaje de Reducción de Complejidad**: **99.26%** de ahorro en comparaciones.
  - **Cobertura de Entidades**: **100.0%** (se conservan todas las entidades únicas originales en al menos una comparación).

---

## 4. Comparación de Embeddings: TF-IDF vs OpenAI
Se construyó una muestra balanceada de **40 pares** (20 positivos y 20 negativos) para comparar la capacidad de separación semántica de **OpenAI embeddings** contra **TF-IDF**:

| Backend | ROC-AUC | PR-AUC | Similitud Media | Desviación Estándar |
|---|---|---|---|---|
| **TF-IDF** | **0.8175** | 0.8055 | 0.3985 | 0.2664 |
| **OpenAI** | 0.8100 | **0.8081** | 0.4464 | 0.1821 |

* **Conclusión**: Ambos motores demostraron una excelente capacidad de separación, pero **TF-IDF** (caracteres n-gramas de 1 a 3) fue seleccionado como el backend principal por defecto debido a que arrojó una métrica de **ROC-AUC superior (0.8175)**, además de que evita la dependencia de red y los costos asociados al API de OpenAI.

---

## 5. Umbrales de Decisión y Cola de Revisión
A partir de la distribución de la similitud coseno calculada masivamente (26,463 variantes únicas), se establecieron umbrales basados en percentiles:
- **Alta Confianza (`high`)**: Similitud coseno $\ge 0.87$. Candidatos para expansión automática de consultas. (10.0% de los pares).
- **Media Confianza (`medium`)**: Similitud coseno entre $0.71$ y $0.87$. Enviados a la cola de revisión manual priorizada. (15.0% de los pares).
- **Baja Confianza (`low`)**: Similitud coseno $< 0.71$. Descartados. (75.0% de los pares).
* **Cola de Revisión**: La cola se ordena asignando prioridad `high` a los pares con similitud coseno $\ge 0.81$ ($medium\_min + 0.10$), concentrando el esfuerzo del analista humano en las discrepancias más cercanas y de mayor valor de decisión.
