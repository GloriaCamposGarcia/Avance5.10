def generate_name_variants(name_base: str, entity_type: str) -> list:
    """
    Genera variantes de nombres a partir de un nombre base normalizado utilizando 
    hasta 7 reglas específicas orientadas a ampliar la cobertura en búsquedas OSINT.
    """
    tokens = [t for t in str(name_base).split() if t]
    if not tokens:
        return []

    variants = []

    def add_variant(v, rule):
        v = ' '.join(str(v).split())
        if v and v not in {x['variant'] for x in variants}:
            variants.append({'variant': v, 'rule': rule})

    # 1) Base canónica
    add_variant(' '.join(tokens), 'base_canonical')

    # 2) Orden de token invertido
    if len(tokens) >= 2:
        add_variant(' '.join(tokens[::-1]), 'reverse_order')

    # 3) Iniciales completas
    initials = ' '.join(t[0] for t in tokens if t)
    add_variant(initials, 'initials_full')

    # 4) Primer token + último token
    if len(tokens) >= 2:
        add_variant(f"{tokens[0]} {tokens[-1]}", 'first_last')

    # 5) Primera inicial + último token
    if len(tokens) >= 2:
        add_variant(f"{tokens[0][0]} {tokens[-1]}", 'first_initial_last')

    # 6) Truncamiento conservando primeros 3 tokens
    if len(tokens) > 3:
        add_variant(' '.join(tokens[:3]), 'truncate_3_tokens')

    # 7) Para MORAL: quitar tokens corporativos residuales comunes
    if str(entity_type).upper() == 'MORAL':
        corp_noise = {'group', 'holding', 'holdings', 'international', 'global', 'trading'}
        filtered = [t for t in tokens if t not in corp_noise]
        if len(filtered) >= 1:
            add_variant(' '.join(filtered), 'moral_remove_corp_noise')

    return variants
