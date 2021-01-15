
class ModComp:
    Mods = {
        'Carbamidomethyl (C)': {'C': 2, 'N': 1, 'O': 1, 'H': 3},
        'Oxidation (M)': {'O': 1},
        'Phospho (STY)': {'H': 1, 'P': 1, 'O': 3},
        'Acetyl (Protein N-term)': {'C': 2, 'O': 1, 'H': 2},
        'Deamidation (NQ)': {'O': 1, 'N': -1, 'H': -1},
        'Methyl (KR)': {'C': 1, 'H': 2},
        'Dimethyl (KR)': {'C': 2, 'H': 4},

    }

    NeutralLoss = {
        'H2O': {'H': 2, 'O': 1}
    }


class BasicModification:
    # Pre-defined mods, where key means query mod and value is used for assembling queried mod
    StandardMods = {
        'Carbamidomethyl': 'Carbamidomethyl',
        'Oxidation': 'Oxidation',
        'Phospho': 'Phospho',
        'Acetyl': 'Acetyl'
    }

    # To extend the query space. Each mod has its alias and itself for quering
    __ModAliasList = {
        'Carbamidomethyl': ['Carbamidomethyl', 'Carbamid', 'Carb', 'Carbamidomethyl[C]'],
        'Oxidation': ['Oxidation', 'Oxi', 'Ox', 'Oxidation[M]'],
        'Phospho': ['Phospholation', 'Phospho', 'Phos', ],
        'Acetyl': ['_[Acetyl (Protein N-term)]'],
    }
    ModAliasDict = {}
    for standard, aliases in __ModAliasList.items():
        for alias in aliases:
            ModAliasDict[alias] = standard
    for alias in list(ModAliasDict.keys()):
        ModAliasDict[alias.upper()] = ModAliasDict[alias]
        ModAliasDict[alias.lower()] = ModAliasDict[alias]

    # Mod rule. This defines the method for mod assembly
    StandardModRule = r'[{mod} ({aa})]'
    ModRuleDict = {'standard': StandardModRule,
                   }

    ModAA = {
        'Carbamidomethyl': ['C'],
        'Oxidation': ['M'],
        'Phospho': ['S', 'T', 'Y'],
    }

    ExtendModAA = ModAA.copy()
    ExtendModAA['Phospho'].extend(['H', 'R', 'K', 'D', 'G', 'M', 'V', 'P', 'N', 'A'])
