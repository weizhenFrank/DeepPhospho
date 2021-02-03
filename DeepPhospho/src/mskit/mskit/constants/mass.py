
class ElementMass:
    Mono = {
        'C': 12.,
        'H': 1.0078250321,
        'O': 15.9949146221,
        'N': 14.0030740052,
        'P': 30.97376151,
        'S': 31.97207069,
        'Na': 22.98976967,
        'Cl': 34.96885271,
        'Ca': 39.9625912,
        'Fe': 55.9349421,
        'Cu': 62.9296011,
    }

    Aver = {

    }


class CompoundMass:
    CompoundMass = {
        'H2O': ElementMass.Mono['H'] * 2 + ElementMass.Mono['O'],
        'NH3': ElementMass.Mono['N'] + ElementMass.Mono['H'] * 3,
        'H3PO4': ElementMass.Mono['H'] * 3 + ElementMass.Mono['P'] + ElementMass.Mono['O'] * 4,
        'HPO3': ElementMass.Mono['H'] + ElementMass.Mono['P'] + ElementMass.Mono['O'] * 3,
        'C2H2O': ElementMass.Mono['C'] * 2 + ElementMass.Mono['H'] * 2 + ElementMass.Mono['O'],
    }


class Mass:
    ResMass = {'A': 71.0371138,
               'C_': 103.00918,
               'C': 160.0306481,
               'D': 115.0269429,
               'E': 129.042593,
               'F': 147.0684139,
               'G': 57.0214637,
               'H': 137.0589118,
               'I': 113.0840639,
               'K': 128.094963,
               'L': 113.0840639,
               'M': 131.0404846,
               'N': 114.0429274,
               'P': 97.0527638,
               'Q': 128.0585774,
               'R': 156.101111,
               'S': 87.0320284,
               'T': 101.0476784,
               'V': 99.0684139,
               'W': 186.0793129,
               'Y': 163.0633285
               }

    ProtonMass = 1.0072766  # H+
    IsotopeMass = 1.003

    ModMass = {'Carbamidomethyl': 57.0214637,
               'C[Carbamidomethyl]': 57.0214637,
               'Oxidation': ElementMass.Mono['O'],
               'M[Oxidation]': ElementMass.Mono['O'],
               'Phospho': CompoundMass.CompoundMass['HPO3'],
               'Acetyl': CompoundMass.CompoundMass['C2H2O'],
               'TMT': 229.1629,
               '1': 147.0353992,  # M[16]
               '2': 167.03203,  # S[80]
               '3': 181.04768,  # T[80]
               '4': 243.06333,  # Y[80]
               }

    ModLossMass = {
        'H3PO4': CompoundMass.CompoundMass['H3PO4']
    }

# C[Carbamidomethyl] = 103.00918 + 57.0214637
# M[Oxidation] = 131.04048 + 16

# H2O + H+ -> 'e'
# H+ -> 'h'
