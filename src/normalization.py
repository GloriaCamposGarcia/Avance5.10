import re
import unicodedata
import pandas as pd

# Patrones para remoción de sufijos corporativos
LEGAL_SUFFIX_PATTERNS = [
    r'\bS\.?A\.?\b', r'\bS\.?\s?DE\s?R\.?L\.?\b', r'\bS\.?\s?DE\s?C\.?V\.?\b',
    r'\bLLC\b', r'\bL\.?L\.?C\.?\b', r'\bLTD\b', r'\bLIMITED\b', r'\bINC\b',
    r'\bCORP\b', r'\bCORPORATION\b', r'\bCOMPANY\b', r'\bCO\.?\b', r'\bHOLDINGS?\b'
]

def strip_accents(text: str) -> str:
    """Remueve acentos y diacríticos de una cadena de texto."""
    text = unicodedata.normalize('NFKD', text)
    return ''.join(ch for ch in text if not unicodedata.combining(ch))

def normalize_whitespace(text: str) -> str:
    """Estandariza los espacios en blanco eliminando duplicados y extremos."""
    return re.sub(r'\s+', ' ', text).strip()

def normalize_entity_name(name: str, entity_type: str) -> dict:
    """
    Normaliza el nombre de una entidad física o moral según las reglas del negocio:
    - Limpia caracteres unicode y acentos.
    - Remueve puntuación no informativa conservando iniciales (ej. r. a.).
    - Para personas morales, remueve sufijos legales comunes de la base canónica.
    - Extrae metadatos (iniciales, conteo de tokens y caracteres).
    """
    raw = '' if pd.isna(name) else str(name)
    x = raw.lower()
    x = strip_accents(x)

    # Conservar iniciales como tokens (r. a.) y eliminar ruido de puntuación no informativa
    x = re.sub(r'[^a-z0-9\s\.]', ' ', x)
    x = re.sub(r'\.(?=\s|$)', ' ', x)  # punto final de abreviatura -> separador
    x = normalize_whitespace(x)

    et = '' if pd.isna(entity_type) else str(entity_type).upper()
    base = x

    removed_suffixes = []
    if et == 'MORAL':
        for pat in LEGAL_SUFFIX_PATTERNS:
            if re.search(pat, base, flags=re.IGNORECASE):
                removed_suffixes.append(pat)
                base = re.sub(pat, ' ', base, flags=re.IGNORECASE)
        base = normalize_whitespace(base)

    tokens = base.split()
    initials = ''.join(tok[0] for tok in tokens if len(tok) > 0)

    return {
        'name_raw': raw,
        'name_norm': x,
        'name_base': base,
        'name_tokens': tokens,
        'name_token_count': len(tokens),
        'name_char_count': len(base.replace(' ', '')),
        'name_initials': initials,
        'removed_legal_suffix_pattern_count': len(removed_suffixes),
    }
