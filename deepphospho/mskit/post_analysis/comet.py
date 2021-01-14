
"""
    Comet style (modifications with symbols annotated):
        S@/T@/Y@ for Phospho (STY)
        M* for Oxidation (M)
        n# for Acetyl (N-term)
        Example: n#DFM*SPKFS@LT@DVEY@PAWCQDDEVPITM*QEIR
"""


def comet_to_intseq(x):
    x = x.replace('_', '')

    x = x.replace('M*', '1')
    x = x.replace('S@', '2')
    x = x.replace('T@', '3')
    x = x.replace('Y@', '4')

    if 'n#' in x:
        x = x.replace('n#', '')
        x = '*' + x
    else:
        x = '@' + x

    return x
