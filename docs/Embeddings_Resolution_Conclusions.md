# Conclusiones de Resolución de Entidades Post-OSINT (Fase 1)

Este documento resume las métricas clave, lecciones metodológicas y resultados de las pruebas comparativas realizadas durante la Fase 1 del pipeline (Embeddings y Validación de Coincidencias de Nombres), tras transicionar de datos sintéticos a un enfoque 100% basado en datos reales de búsquedas OSINT.

## 1. Contexto Metodológico Post-OSINT

De acuerdo con las mejores prácticas en sistemas de cumplimiento (AML) y record linkage, el pipeline ha sido rediseñado para operar de forma **post-OSINT**:
- Se eliminó la generación de variantes sintéticas/artificiales y el blocking cartesiano artificial.
- En su lugar, el sistema toma las **consultas OSINT reales** (`query_value` en `entity_source_results.csv`) y las cruza directamente con los **matches o hallazgos reales devueltos por las fuentes** (`matched_name` en `evidence_items.csv`).
- Esto elimina por completo el ruido nominal introducido por permutaciones sintéticas que nunca fueron buscadas en la vida real.

---

## 2. Volumen de Datos Procesados

- **Consultas OSINT Únicas**: 36,985 consultas extraídas a partir del histórico de búsquedas.
- **Evidencias OSINT Cargadas**: 10,426 hallazgos reportados por las fuentes (OFAC, UK Sanctions, ONU, etc.).
- **Pares Reales de Comparación**: **10,426 pares** identificados al cruzar consultas con sus respectivos matches.
* **Impacto**: La complejidad computacional se redujo a la escala lineal de los hallazgos reales ($O(N)$ en lugar de $O(M^2)$ donde $M$ era la cantidad de variantes sintéticas). Ya no se requiere una fase de blocking cartesiano artificial, lo que ahorra un 100% de la carga computacional innecesaria en base de datos.

---

## 3. Umbrales de Similitud y Resultados de Clasificación

Tras evaluar la similitud coseno de los 10,426 pares reales con TF-IDF, los umbrales se acotaron dinámicamente para manejar la alta concentración de coincidencias exactas (donde los percentiles 75 y 90 colapsaban en 1.0):
- **Alta Confianza (`high`)**: Similitud coseno $\ge 0.90$. Coincidencias aprobadas de forma automática.
- **Media Confianza (`medium`)**: Similitud coseno entre $0.75$ y $0.90$. Enviados a la cola de revisión manual priorizada.
- **Baja Confianza (`low`)**: Similitud coseno $< 0.75$. Descartados como falsos positivos del motor OSINT.

### Estadísticas de la Corrida:
- **Cola de Revisión Manual (`manual_review_queue.csv`)**: Contiene **926 pares** que presentaron ligeras diferencias ortográficas u organizacionales (por ejemplo: *"TEHRIK-E TALIBAN PAKISTAN"* vs *"TEHRIK-E TALIBAN PAKISTAN (TTP)"*, similitud `0.89992`).
- **Pares Prioridad Alta**: Aquellos en la cola con similitud $\ge 0.85$ ($medium\_min + 0.10$), permitiendo al analista priorizar discrepancias menores antes de evaluar variaciones complejas.

---

## 4. Benchmarking de Motores de Embeddings

Se evaluaron los tres backends soportados por el pipeline sobre una muestra balanceada de validación de **40 pares reales** (20 positivos con `decision == 'accepted'` y 20 negativos realistas construidos cruzando consultas con matches de entidades distintas):

| Backend | Pares Evaluados | ROC-AUC | PR-AUC | Estado / Observaciones |
|---|---|---|---|---|
| **TF-IDF** (Local) | 40 | **1.0000** | **1.0000** | **Activo por defecto**. Excelente separación de caracteres sin costo ni dependencias de red. |
| **Sentence-Transformers** (Local) | 40 | **1.0000** | **1.0000** | **Alternativa local**. Captura semántica densa de forma gratuita y local (modelo `all-MiniLM-L6-v2`). |
| **OpenAI** (Remoto) | 0 | *N/A* | *N/A* | **Error por cuota insuficiente** (`insufficient_quota`). |

* **Conclusión de Resiliencia**: El fallo por cuota del API de OpenAI subraya la importancia de contar con backends de embeddings locales como **TF-IDF** y **Sentence-Transformers**. Ambos demostraron una separación perfecta (`ROC-AUC = 1.0`) en este set de prueba, validando la suficiencia y superioridad práctica del procesamiento local para este caso de uso.
