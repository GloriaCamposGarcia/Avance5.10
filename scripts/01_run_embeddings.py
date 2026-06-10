import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
try:
    from dotenv import load_dotenv
    HAS_DOTENV = True
except ImportError:
    HAS_DOTENV = False

# Añadir la raíz del proyecto al path para poder importar de src
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from src.normalization import normalize_entity_name
from src.embeddings import (
    build_embeddings, _rowwise_cosine, classify_similarity_level,
    _score_pairs_for_backend, _safe_auc, _safe_ap
)

def main():
    # Intentar cargar variables de entorno desde un archivo .env local
    if HAS_DOTENV:
        try:
            load_dotenv(dotenv_path=project_root / '.env')
        except Exception:
            pass

    # 1. Definir rutas y cargar datasets
    raw_data_path = project_root / 'data' / 'raw' / 'entity_source_results.csv'
    evidence_data_path = project_root / 'data' / 'raw' / 'evidence_items.csv'
    processed_dir = project_root / 'data' / 'processed'
    processed_dir.mkdir(parents=True, exist_ok=True)

    if not raw_data_path.exists():
        print(f"ERROR: No se encontró el dataset de resultados en {raw_data_path}")
        print("Por favor, coloca 'entity_source_results.csv' en la carpeta data/raw/")
        sys.exit(1)

    if not evidence_data_path.exists():
        print(f"ERROR: No se encontró el dataset de evidencias en {evidence_data_path}")
        print("Por favor, coloca 'evidence_items.csv' en la carpeta data/raw/")
        sys.exit(1)

    print(f"Cargando dataset de resultados de búsqueda desde {raw_data_path}...")
    df_raw = pd.read_csv(raw_data_path)
    
    # Extraer catálogo de consultas reales desde query_used
    print("Extrayendo catálogo de consultas OSINT reales desde la columna 'query_used'...")
    queries = []
    import ast
    for val in df_raw['query_used'].dropna().unique():
        try:
            data = ast.literal_eval(val)
            if isinstance(data, list) and len(data) > 0:
                item = data[0]
                ent_id = item.get('entity_id')
                name = item.get('query_value')
                metadata = item.get('metadata', {})
                country = metadata.get('country_code', '')
                raw_type = metadata.get('entity_type', '')
                ent_type = 'MORAL' if raw_type == 'PM' else ('FISICA' if raw_type == 'PF' else raw_type)
                
                # Extraemos a nivel de entity_id y source_id (desde query_used)
                source_id = item.get('source_id', '')
                if ent_id and name:
                    queries.append({
                        'entity_id': ent_id,
                        'source_id': source_id,
                        'query_value': name,
                        'entity_type': ent_type,
                        'country_code': country
                    })
        except Exception:
            continue
            
    df_queries = pd.DataFrame(queries)
    if not df_queries.empty:
        df_queries = df_queries.drop_duplicates(subset=['entity_id', 'source_id'])
    print(f"Consultas únicas extraídas: {len(df_queries)}")

    print(f"Cargando dataset de evidencias desde {evidence_data_path}...")
    df_evidences = pd.read_csv(evidence_data_path)
    print(f"Evidencias cargadas: {len(df_evidences)}")

    # 2. Cruzar consultas y evidencias para pares reales
    print("\nCruzando consultas con evidencias encontradas por entity_id y source_id...")
    df_evidences = df_evidences.dropna(subset=['entity_id', 'source_id', 'matched_name'])
    
    df_pairs = pd.merge(
        df_evidences[['evidence_id', 'entity_id', 'source_id', 'matched_name', 'decision', 'identity_score', 'review_required']],
        df_queries[['entity_id', 'source_id', 'query_value', 'entity_type', 'country_code']],
        on=['entity_id', 'source_id'],
        how='inner'
    )
    print(f"Pares de comparación reales identificados: {len(df_pairs)}")
    if df_pairs.empty:
        print("ERROR: No se encontraron coincidencias entre las consultas y las evidencias.")
        sys.exit(1)

    # 3. Normalización de nombres de consulta y match
    print("\nEjecutando normalización de nombres base (consultas y matches)...")
    df_pairs['query_base'] = df_pairs.apply(
        lambda r: normalize_entity_name(r['query_value'], r['entity_type'])['name_base'] if pd.notna(r['query_value']) else '',
        axis=1
    )
    df_pairs['matched_base'] = df_pairs.apply(
        lambda r: normalize_entity_name(r['matched_name'], r['entity_type'])['name_base'] if pd.notna(r['matched_name']) else '',
        axis=1
    )
    df_pairs = df_pairs[(df_pairs['query_base'] != '') & (df_pairs['matched_base'] != '')].copy()
    print(f"Pares válidos tras normalización: {len(df_pairs)}")

    # 4. Inicialización de embeddings (TF-IDF)
    print("\nInicializando representaciones de embeddings (TF-IDF)...")
    unique_texts = pd.Index(
        pd.concat([df_pairs['query_base'], df_pairs['matched_base']], ignore_index=True)
        .dropna().unique()
    )
    text_to_idx = {t: i for i, t in enumerate(unique_texts)}
    X_var, vectorizer, embedding_backend = build_embeddings(unique_texts, backend='tfidf')
    print(f"Embeddings inicializados con backend: {embedding_backend}. Dimensión: {X_var.shape}")

    ia = df_pairs['query_base'].map(text_to_idx).to_numpy()
    ib = df_pairs['matched_base'].map(text_to_idx).to_numpy()
    valid = (~pd.isna(ia)) & (~pd.isna(ib))
    df_pairs = df_pairs.loc[valid].copy()
    ia = ia[valid].astype(int)
    ib = ib[valid].astype(int)

    # 5. Cálculo vectorizado de similitud coseno
    print("Calculando similitud coseno en lotes...")
    batch_size = 100_000
    scores = np.empty(len(df_pairs), dtype=float)
    for start in range(0, len(df_pairs), batch_size):
        end = min(start + batch_size, len(df_pairs))
        A = X_var[ia[start:end]]
        B = X_var[ib[start:end]]
        scores[start:end] = _rowwise_cosine(A, B)

    df_pairs['cosine_similarity'] = np.round(scores, 6)

    # 6. Definición de umbrales y colas de decisión (acotados para evitar colapso en datos con muchos matches exactos)
    q75 = float(df_pairs['cosine_similarity'].quantile(0.75))
    q90 = float(df_pairs['cosine_similarity'].quantile(0.90))

    thresholds = {
        'high_min': round(max(min(q90, 0.90), 0.80), 6),
        'medium_min': round(max(min(q75, 0.75), 0.60), 6),
    }
    print(f"\nUmbrales establecidos - high_min: {thresholds['high_min']}, medium_min: {thresholds['medium_min']}")

    df_pairs['confidence_level'] = df_pairs['cosine_similarity'].apply(
        lambda s: classify_similarity_level(s, thresholds['high_min'], thresholds['medium_min'])
    )
    df_pairs['requires_manual_review'] = df_pairs['confidence_level'].eq('medium')
    df_pairs['auto_expand_query'] = df_pairs['confidence_level'].eq('high')

    # Guardar resultados
    similarity_out_path = processed_dir / 'similarity_results.csv'
    df_pairs = df_pairs.sort_values('cosine_similarity', ascending=False)
    
    keep_cols = [
        'evidence_id', 'entity_id', 'source_id', 'query_value', 'matched_name',
        'query_base', 'matched_base', 'country_code', 'entity_type', 
        'cosine_similarity', 'confidence_level', 'requires_manual_review', 'decision'
    ]
    keep_cols = [col for col in keep_cols if col in df_pairs.columns]
    df_pairs[keep_cols].to_csv(similarity_out_path, index=False)
    print(f"Resultados de similitud guardados en: {similarity_out_path}")

    # Generación de cola de revisión manual
    review_queue = df_pairs[df_pairs['requires_manual_review']].copy()
    review_queue['priority'] = np.where(
        review_queue['cosine_similarity'] >= (thresholds['medium_min'] + 0.10),
        'high',
        'medium'
    )
    review_queue = review_queue.sort_values(['priority', 'cosine_similarity'], ascending=[True, False])
    review_out_path = processed_dir / 'manual_review_queue.csv'
    review_queue[keep_cols + ['priority']].to_csv(review_out_path, index=False)
    print(f"Cola de revisión manual guardada en: {review_out_path}")

    # 7. Comparación OpenAI vs TF-IDF vs Sentence-Transformers (Muestra de validación con datos reales)
    print("\nEjecutando validación comparativa con datos reales (OpenAI vs TF-IDF vs Sentence-Transformers)...")
    positives = df_pairs[df_pairs['decision'] == 'accepted'].copy()
    positives['same_entity'] = 1

    negatives = []
    rng = np.random.default_rng(42)
    matched_pool = df_pairs['matched_name'].unique()
    
    if len(df_pairs) > 0 and len(matched_pool) > 1:
        sampled_queries = df_pairs.sample(n=min(100, len(df_pairs)), random_state=42)
        for _, row in sampled_queries.iterrows():
            different_matches = df_pairs[df_pairs['entity_id'] != row['entity_id']]['matched_name'].unique()
            if len(different_matches) > 0:
                neg_match = rng.choice(different_matches)
                negatives.append({
                    'query_value': row['query_value'],
                    'matched_name': neg_match,
                    'entity_id_a': row['entity_id'],
                    'entity_id_b': 'NEGATIVE_SAMPLE',
                    'same_entity': 0
                })
                
    df_neg = pd.DataFrame(negatives)
    sample_size = min(20, len(positives), len(df_neg))
    
    if sample_size > 0:
        pos_sample = positives[['query_value', 'matched_name', 'entity_id', 'same_entity']].rename(
            columns={'entity_id': 'entity_id_a'}
        ).sample(n=sample_size, random_state=42)
        pos_sample['entity_id_b'] = pos_sample['entity_id_a']
        
        neg_sample = df_neg.sample(n=sample_size, random_state=42)
        validation_sample = pd.concat([pos_sample, neg_sample], ignore_index=True)
        validation_sample = validation_sample.rename(columns={
            'query_value': 'variant_a',
            'matched_name': 'variant_b'
        })

        results = []
        for backend_name in ['openai', 'tfidf', 'sentence-transformers']:
            try:
                scored = _score_pairs_for_backend(validation_sample, backend_name)
                results.append({
                    'backend': scored['backend'].iloc[0],
                    'pairs_scored': int(len(scored)),
                    'roc_auc': round(_safe_auc(scored['same_entity'], scored['cosine_similarity']), 4),
                    'pr_auc': round(_safe_ap(scored['same_entity'], scored['cosine_similarity']), 4),
                    'error': np.nan
                })
            except Exception as exc:
                results.append({
                    'backend': backend_name,
                    'pairs_scored': 0,
                    'roc_auc': np.nan,
                    'pr_auc': np.nan,
                    'error': str(exc)
                })
        print(pd.DataFrame(results).to_string(index=False))
    else:
        print("No hay suficientes pares reales aceptados/negativos para validación.")

    print("\nFase 1 completada con éxito.")

if __name__ == '__main__':
    main()
