import pandas as pd

def build_blocking_candidates(variants_df: pd.DataFrame) -> pd.DataFrame:
    """
    Realiza la etapa de blocking sobre el conjunto de variantes generadas:
    - Agrega metadatos (cantidad de tokens, primera letra y longitud de caracteres sin espacios).
    - Remueve variantes nulas o vacías.
    - Hace un merge cruzado (inner join) basado en 'entity_type', 'country_code' y 'first_char'.
    - Elimina auto-comparaciones (mismo entity_id) y duplicados simétricos (a < b).
    - Aplica filtros de control en diferencia de longitud de caracteres (<= 15) y tokens (<= 2).
    - Retorna el DataFrame de pares candidatos listos para similitud.
    """
    blk_df = variants_df.copy()
    blk_df['variant_tokens'] = blk_df['name_variant'].fillna('').str.split().str.len()
    blk_df['variant_first_char'] = blk_df['name_variant'].fillna('').str.strip().str[0].fillna('')
    blk_df['variant_len_chars'] = blk_df['name_variant'].fillna('').str.replace(' ', '', regex=False).str.len()

    blk_df = blk_df[blk_df['name_variant'].notna() & (blk_df['name_variant'].str.len() > 0)].copy()

    # Renombrar columnas para la parte izquierda de la comparación
    left = blk_df.rename(columns={
        'entity_id': 'entity_id_a',
        'entity_type': 'entity_type_a',
        'country_code': 'country_code_a',
        'name_variant': 'variant_a',
        'variant_rule': 'rule_a',
        'variant_tokens': 'tokens_a',
        'variant_first_char': 'first_char_a',
        'variant_len_chars': 'len_chars_a'
    })

    # Renombrar columnas para la parte derecha de la comparación
    right = blk_df.rename(columns={
        'entity_id': 'entity_id_b',
        'entity_type': 'entity_type_b',
        'country_code': 'country_code_b',
        'name_variant': 'variant_b',
        'variant_rule': 'rule_b',
        'variant_tokens': 'tokens_b',
        'variant_first_char': 'first_char_b',
        'variant_len_chars': 'len_chars_b'
    })

    # Merge cartesiano por criterios de indexación de blocking
    cand = left.merge(
        right, 
        how='inner', 
        left_on=['entity_type_a', 'country_code_a', 'first_char_a'], 
        right_on=['entity_type_b', 'country_code_b', 'first_char_b']
    )

    # Evitar auto-pares y duplicados simétricos
    cand = cand[cand['entity_id_a'] < cand['entity_id_b']].copy()

    # Restringir por diferencia en la cantidad de tokens
    cand = cand[(cand['tokens_a'] - cand['tokens_b']).abs() <= 2].copy()

    # Restringir por diferencia de longitud en caracteres
    cand = cand[(cand['len_chars_a'] - cand['len_chars_b']).abs() <= 15].copy()

    # Generar un ID único del par y eliminar duplicados redundantes
    cand['pair_id'] = (
        cand['entity_id_a'].astype(str) + '||' + 
        cand['entity_id_b'].astype(str) + '||' + 
        cand['variant_a'].astype(str) + '||' + 
        cand['variant_b'].astype(str)
    )
    cand = cand.drop_duplicates(subset=['pair_id']).copy()

    return cand
