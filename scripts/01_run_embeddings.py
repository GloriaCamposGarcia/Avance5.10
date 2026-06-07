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
from src.variants import generate_name_variants
from src.blocking import build_blocking_candidates
from src.embeddings import (
    build_embeddings, _rowwise_cosine, classify_similarity_level,
    _make_positive_pairs, _make_negative_pair, _score_pairs_for_backend,
    _safe_auc, _safe_ap
)

def main():
    # Intentar cargar variables de entorno desde un archivo .env local
    if HAS_DOTENV:
        try:
            load_dotenv(dotenv_path=project_root / '.env')
        except Exception:
            pass

    # 1. Definir rutas y cargar dataset
    raw_data_path = project_root / 'data' / 'raw' / 'entity_source_results.csv'
    processed_dir = project_root / 'data' / 'processed'
    processed_dir.mkdir(parents=True, exist_ok=True)

    if not raw_data_path.exists():
        print(f"ERROR: No se encontró el dataset en {raw_data_path}")
        print("Por favor, coloca 'entity_source_results.csv' en la carpeta data/raw/")
        sys.exit(1)

    print(f"Cargando dataset desde {raw_data_path}...")
    df_raw = pd.read_csv(raw_data_path)
    
    # Extraer catálogo de entidades únicas desde query_used
    print("Extrayendo catálogo de entidades únicas desde la columna 'query_used'...")
    entities = {}
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
                
                if ent_id and name and ent_id not in entities:
                    entities[ent_id] = {
                        'entity_id': ent_id,
                        'entity_name': name,
                        'entity_type': ent_type,
                        'country_code': country
                    }
        except Exception:
            continue
            
    df = pd.DataFrame(list(entities.values()))
    print(f"Dataset de entidades construido con dimensiones: {df.shape}")

    # 2. Normalización de nombres de entidades
    print("\nEjecutando normalización de nombres...")
    norm_rows = []
    for row in df[['entity_id', 'entity_name', 'entity_type', 'country_code']].itertuples(index=False):
        d = normalize_entity_name(row.entity_name, row.entity_type)
        d['entity_id'] = row.entity_id
        d['entity_type'] = row.entity_type
        d['country_code'] = row.country_code
        norm_rows.append(d)
    df_norm = pd.DataFrame(norm_rows)
    print(f"Normalización completa. Cardinalidad única de nombres base: {df_norm['name_base'].nunique()} de {len(df_norm)}")

    # 3. Generación de variantes de nombres
    print("\nGenerando variantes de nombres...")
    variant_rows = []
    for r in df_norm[['entity_id', 'entity_type', 'country_code', 'name_raw', 'name_base']].itertuples(index=False):
        vars_ = generate_name_variants(r.name_base, r.entity_type)
        for item in vars_:
            variant_rows.append({
                'entity_id': r.entity_id,
                'entity_type': r.entity_type,
                'country_code': r.country_code,
                'name_raw': r.name_raw,
                'name_base': r.name_base,
                'name_variant': item['variant'],
                'variant_rule': item['rule']
            })
    variants_df = pd.DataFrame(variant_rows)
    print(f"Variantes generadas: {len(variants_df)}")

    # 4. Blocking para reducción de pares
    print("\nEjecutando blocking de candidatos...")
    cand = build_blocking_candidates(variants_df)
    print(f"Pares bloqueados resultantes: {len(cand)}")

    # 5. Inicialización de embeddings y cálculo de similitud (TF-IDF)
    print("\nInicializando representaciones de embeddings (TF-IDF)...")
    unique_variants = pd.Index(variants_df['name_variant'].dropna().unique())
    variant_to_idx = {v: i for i, v in enumerate(unique_variants)}
    X_var, vectorizer, embedding_backend = build_embeddings(unique_variants, backend='tfidf')
    print(f"Embeddings inicializados con backend: {embedding_backend}. Dimensión: {X_var.shape}")

    # Filtrar candidatos rápidos (segundo carácter y diferencias menores)
    cand_fast = cand.copy()
    if {'tokens_a', 'tokens_b'}.issubset(cand_fast.columns):
        cand_fast = cand_fast[(cand_fast['tokens_a'] - cand_fast['tokens_b']).abs() <= 1]
    if {'len_chars_a', 'len_chars_b'}.issubset(cand_fast.columns):
        cand_fast = cand_fast[(cand_fast['len_chars_a'] - cand_fast['len_chars_b']).abs() <= 8]

    cand_fast['second_char_a'] = cand_fast['variant_a'].fillna('').str.replace(' ', '', regex=False).str[1:2]
    cand_fast['second_char_b'] = cand_fast['variant_b'].fillna('').str.replace(' ', '', regex=False).str[1:2]
    cand_fast = cand_fast[cand_fast['second_char_a'] == cand_fast['second_char_b']]

    print(f"Pares tras filtro rápido de blocking: {len(cand_fast)}")

    ia = cand_fast['variant_a'].map(variant_to_idx).to_numpy()
    ib = cand_fast['variant_b'].map(variant_to_idx).to_numpy()
    valid = (~pd.isna(ia)) & (~pd.isna(ib))
    cand_fast = cand_fast.loc[valid].copy()
    ia = ia[valid].astype(int)
    ib = ib[valid].astype(int)

    # Cálculo vectorizado por bloques
    print("Calculando similitud coseno en lotes...")
    batch_size = 100_000
    scores = np.empty(len(cand_fast), dtype=float)
    for start in range(0, len(cand_fast), batch_size):
        end = min(start + batch_size, len(cand_fast))
        A = X_var[ia[start:end]]
        B = X_var[ib[start:end]]
        scores[start:end] = _rowwise_cosine(A, B)

    keep_cols = ['pair_id', 'entity_id_a', 'entity_id_b', 'variant_a', 'variant_b', 'country_code_a', 'rule_a', 'rule_b']
    keep_cols = [col for col in keep_cols if col in cand_fast.columns]
    sim_df = cand_fast[keep_cols].copy()
    if 'country_code_a' in sim_df.columns:
        sim_df = sim_df.rename(columns={'country_code_a': 'country_code'})
    sim_df['cosine_similarity'] = np.round(scores, 6)
    sim_df = sim_df.sort_values('cosine_similarity', ascending=False)

    # 6. Definición de umbrales y colas de decisión
    q75 = float(sim_df['cosine_similarity'].quantile(0.75))
    q90 = float(sim_df['cosine_similarity'].quantile(0.90))

    thresholds = {
        'high_min': round(max(q90, 0.80), 6),
        'medium_min': round(max(q75, 0.60), 6),
    }
    print(f"\nUmbrales establecidos - high_min: {thresholds['high_min']}, medium_min: {thresholds['medium_min']}")

    sim_scored = sim_df.copy()
    sim_scored['confidence_level'] = sim_scored['cosine_similarity'].apply(
        lambda s: classify_similarity_level(s, thresholds['high_min'], thresholds['medium_min'])
    )
    sim_scored['requires_manual_review'] = sim_scored['confidence_level'].eq('medium')
    sim_scored['auto_expand_query'] = sim_scored['confidence_level'].eq('high')

    # Guardar similitudes
    similarity_out_path = processed_dir / 'similarity_results.csv'
    sim_scored.to_csv(similarity_out_path, index=False)
    print(f"Resultados de similitud guardados en: {similarity_out_path}")

    # Generación de cola de revisión manual
    review_queue = sim_scored[sim_scored['requires_manual_review']].copy()
    review_queue['priority'] = np.where(
        review_queue['cosine_similarity'] >= (thresholds['medium_min'] + 0.10),
        'high',
        'medium'
    )
    review_queue = review_queue.sort_values(['priority', 'cosine_similarity'], ascending=[True, False])
    review_out_path = processed_dir / 'manual_review_queue.csv'
    review_queue.to_csv(review_out_path, index=False)
    print(f"Cola de revisión manual guardada en: {review_out_path}")

    # 7. Comparación OpenAI vs TF-IDF (Muestra de validación)
    print("\nEjecutando validación comparativa (OpenAI vs TF-IDF)...")
    variant_pool = variants_df[['entity_id', 'name_variant'] + [c for c in ['country_code', 'entity_type'] if c in variants_df.columns]].copy()
    variant_pool = variant_pool[variant_pool['name_variant'].notna() & (variant_pool['name_variant'].astype(str).str.len() > 0)].copy()
    variant_pool['name_variant'] = variant_pool['name_variant'].astype(str)

    rng = np.random.default_rng(42)
    entity_groups = {entity_id: group for entity_id, group in variant_pool.groupby('entity_id')}
    positive_entity_ids = [entity_id for entity_id, group in entity_groups.items() if len(group) >= 2]
    
    sample_per_class = min(20, len(positive_entity_ids))
    if sample_per_class > 0:
        positive_rows = _make_positive_pairs(sample_per_class, positive_entity_ids, entity_groups, rng)
        negative_rows = [_make_negative_pair(variant_pool, entity_groups, rng) for _ in range(sample_per_class)]
        validation_sample = pd.DataFrame(positive_rows + negative_rows).sample(frac=1.0, random_state=42).reset_index(drop=True)

        results = []
        for backend_name in ['openai', 'tfidf']:
            try:
                scored = _score_pairs_for_backend(validation_sample, backend_name)
                results.append({
                    'backend': scored['backend'].iloc[0],
                    'pairs_scored': int(len(scored)),
                    'roc_auc': round(_safe_auc(scored['same_entity'], scored['cosine_similarity']), 4),
                    'pr_auc': round(_safe_ap(scored['same_entity'], scored['cosine_similarity']), 4),
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
        print("No hay suficientes variantes para correr la comparación OpenAI vs TF-IDF.")

    print("\nFase 1 completada con éxito.")

if __name__ == '__main__':
    main()
