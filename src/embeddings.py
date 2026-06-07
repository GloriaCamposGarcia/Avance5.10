import os
import json
import numpy as np
import pandas as pd
from urllib import error, request
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
from sklearn.metrics import roc_auc_score, average_precision_score

# Variables de entorno y defaults de OpenAI
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small").strip()
EMBEDDING_BACKEND = os.getenv("EMBEDDING_BACKEND", "tfidf").strip().lower()

def _normalize_embedding_matrix(matrix):
    """Normaliza L2 filas de la matriz dispersa o densa."""
    if sparse.issparse(matrix):
        return normalize(matrix, norm='l2', axis=1)
    dense = np.asarray(matrix, dtype=float)
    norms = np.linalg.norm(dense, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return dense / norms

def _openai_embeddings_batch(texts):
    """Genera embeddings vía API de OpenAI usando urllib (lote)."""
    api_key = os.getenv("OPENAI_API_KEY", OPENAI_API_KEY)
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY no está configurado. Configure la variable de entorno o use tfidf.")

    payload = json.dumps({'model': OPENAI_EMBEDDING_MODEL, 'input': list(texts)}).encode('utf-8')
    req = request.Request('https://api.openai.com/v1/embeddings', data=payload, method='POST')
    req.add_header('Authorization', f'Bearer {api_key}')
    req.add_header('Content-Type', 'application/json')

    try:
        with request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read().decode('utf-8'))
    except error.HTTPError as exc:
        detail = exc.read().decode('utf-8', errors='ignore') if exc.fp else str(exc)
        raise RuntimeError(f"Error al consultar las incrustaciones de OpenAI: {detail}") from exc

    return [item['embedding'] for item in data['data']]

def build_embeddings(texts, backend=None, batch_size=128):
    """
    Construye la representación de embeddings (matriz CSR o array Numpy L2 normalizado):
    - backend 'openai': consulta el API de OpenAI.
    - backend 'tfidf': entrena TF-IDF local a nivel de caracteres (n-gramas 1-3).
    """
    selected_backend = (backend or EMBEDDING_BACKEND).strip().lower()
    texts_index = pd.Index(texts).dropna().astype(str)

    if selected_backend == 'openai':
        chunks = []
        for start in range(0, len(texts_index), batch_size):
            batch = texts_index[start:start + batch_size].tolist()
            chunks.extend(_openai_embeddings_batch(batch))
        X = _normalize_embedding_matrix(np.asarray(chunks, dtype=float))
        return X, None, 'openai'

    vectorizer = TfidfVectorizer(ngram_range=(1, 3), analyzer='char_wb', min_df=1)
    X = vectorizer.fit_transform(texts_index)
    return X, vectorizer, 'tfidf'

def _rowwise_cosine(A, B):
    """Calcula similitud coseno fila por fila de manera eficiente (normalizado L2)."""
    if sparse.issparse(A):
        return np.asarray(A.multiply(B).sum(axis=1)).ravel()
    return np.sum(np.asarray(A) * np.asarray(B), axis=1)

def top_neighbors_for_variant(query_text, X_var, unique_variants, variant_to_idx, top_k=10):
    """Devuelve los vecinos más cercanos en similitud coseno para una consulta dada."""
    if query_text not in variant_to_idx:
        return pd.DataFrame({'message': [f'Variante no encontrada: {query_text}']})

    q_idx = variant_to_idx[query_text]
    sims = cosine_similarity(X_var[q_idx], X_var).ravel()
    order = sims.argsort()[::-1]
    order = [i for i in order if i != q_idx][:top_k]

    rows = []
    for i in order:
        rows.append({
            'query_variant': query_text,
            'neighbor_variant': unique_variants[i],
            'cosine_similarity': round(float(sims[i]), 6)
        })
    return pd.DataFrame(rows)

def classify_similarity_level(score: float, high_min: float, medium_min: float) -> str:
    """Clasifica un puntaje de similitud coseno en 'high', 'medium' o 'low'."""
    if score >= high_min:
        return 'high'
    if score >= medium_min:
        return 'medium'
    return 'low'

# Funciones de validación comparativa
def _make_positive_pairs(sample_count, positive_entity_ids, entity_groups, rng):
    """Construye pares de la misma entidad (positivos)."""
    rows = []
    chosen_entities = rng.choice(positive_entity_ids, size=sample_count, replace=False)
    for entity_id in chosen_entities:
        group = entity_groups[entity_id].reset_index(drop=True)
        idxs = rng.choice(len(group), size=2, replace=False)
        row_a = group.iloc[int(idxs[0])]
        row_b = group.iloc[int(idxs[1])]
        rows.append({
            'variant_a': row_a['name_variant'],
            'variant_b': row_b['name_variant'],
            'entity_id_a': row_a['entity_id'],
            'entity_id_b': row_b['entity_id'],
            'same_entity': 1,
        })
    return rows

def _make_negative_pair(variant_pool, entity_groups, rng):
    """Construye pares de entidades diferentes (negativos), idealmente del mismo país/tipo."""
    bucket_cols = [c for c in ['country_code', 'entity_type'] if c in variant_pool.columns]
    candidate_df = variant_pool.copy()

    if bucket_cols:
        candidate_df['bucket'] = candidate_df[bucket_cols].astype(str).agg('||'.join, axis=1)
        bucket_values = candidate_df['bucket'].dropna().unique().tolist()
        rng.shuffle(bucket_values)
        for bucket in bucket_values:
            bucket_df = candidate_df[candidate_df['bucket'] == bucket]
            bucket_entities = bucket_df['entity_id'].dropna().unique().tolist()
            if len(bucket_entities) < 2:
                continue
            ent_a, ent_b = rng.choice(bucket_entities, size=2, replace=False)
            row_a = bucket_df[bucket_df['entity_id'] == ent_a].sample(n=1, random_state=int(rng.integers(0, 1_000_000))).iloc[0]
            row_b = bucket_df[bucket_df['entity_id'] == ent_b].sample(n=1, random_state=int(rng.integers(0, 1_000_000))).iloc[0]
            return {
                'variant_a': row_a['name_variant'],
                'variant_b': row_b['name_variant'],
                'entity_id_a': row_a['entity_id'],
                'entity_id_b': row_b['entity_id'],
                'same_entity': 0,
            }

    ent_a, ent_b = rng.choice(list(entity_groups.keys()), size=2, replace=False)
    row_a = entity_groups[ent_a].sample(n=1, random_state=int(rng.integers(0, 1_000_000))).iloc[0]
    row_b = entity_groups[ent_b].sample(n=1, random_state=int(rng.integers(0, 1_000_000))).iloc[0]
    return {
        'variant_a': row_a['name_variant'],
        'variant_b': row_b['name_variant'],
        'entity_id_a': row_a['entity_id'],
        'entity_id_b': row_b['entity_id'],
        'same_entity': 0,
    }

def _score_pairs_for_backend(sample_df, backend_name):
    """Mapea y calcula la similitud coseno para una muestra dada con el backend indicado."""
    texts = pd.Index(pd.concat([sample_df['variant_a'], sample_df['variant_b']], ignore_index=True).dropna().astype(str).unique())
    X_backend, _, actual_backend = build_embeddings(texts, backend=backend_name)
    idx_map = {text: idx for idx, text in enumerate(texts)}

    left_idx = sample_df['variant_a'].map(idx_map).to_numpy()
    right_idx = sample_df['variant_b'].map(idx_map).to_numpy()

    valid = (~pd.isna(left_idx)) & (~pd.isna(right_idx))
    if not valid.all():
        sample_df = sample_df.loc[valid].copy()
        left_idx = left_idx[valid].astype(int)
        right_idx = right_idx[valid].astype(int)

    if sparse.issparse(X_backend):
        scores = np.asarray(X_backend[left_idx].multiply(X_backend[right_idx]).sum(axis=1)).ravel()
    else:
        scores = np.sum(np.asarray(X_backend[left_idx]) * np.asarray(X_backend[right_idx]), axis=1)

    out = sample_df.copy()
    out['backend'] = actual_backend
    out['cosine_similarity'] = np.round(scores, 6)
    return out

def _safe_auc(y_true, scores):
    """Calcula ROC AUC con manejo de excepciones."""
    try:
        return float(roc_auc_score(y_true, scores))
    except Exception:
        return float('nan')

def _safe_ap(y_true, scores):
    """Calcula Average Precision con manejo de excepciones."""
    try:
        return float(average_precision_score(y_true, scores))
    except Exception:
        return float('nan')
