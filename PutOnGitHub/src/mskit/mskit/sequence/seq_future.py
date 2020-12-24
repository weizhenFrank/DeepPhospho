import re
from itertools import combinations

from mskit import rapid_kit


def batch_add_target_mod(pep_list, mod_type: dict = None, mod_processor=None):
    """
    TODO : This may result of some redundant results (dont know why)
    """
    modpep_list = []
    for pep in rapid_kit.drop_list_duplicates(pep_list):
        modpep_list.extend(add_target_mod(pep, mod_type, mod_processor))
    return rapid_kit.drop_list_duplicates(modpep_list)


def batch_add_target_charge(modpep_list, charge=(2, 3)):
    prec_list = []
    for modpep in rapid_kit.drop_list_duplicates(modpep_list):
        prec_list.extend(add_target_charge(modpep, charge))
    return prec_list


def add_target_mod(pep, mod_type: dict = None, mod_processor=None):
    """
    :param pep: target peptide
    :param mod_type: dict like {'Carbamidomethyl': -1, 'Oxidation': 1}
    where -1 means all possible AAs will be modified with the determined modification
    and an integer means the maximum number of this modification
    :param mod_processor: the mod add processor contains the standard mod rule or customed mod rule
    """
    if mod_type:
        mods = []
        for _mod, _num in mod_type.items():
            _standard_mod = mod_processor.get_standard_mod(_mod)
            possible_aa = mod_processor.query_possible_aa(_standard_mod)
            possible_site = sorted([_.end() for one_aa in possible_aa for _ in re.finditer(one_aa, pep)])
            possible_site_num = len(possible_site)
            if _num > possible_site_num:
                _num = possible_site_num
            if _num == 0 or possible_site_num == 0:
                continue
            elif _num == -1:
                mod = [[(_, _standard_mod) for _ in possible_site]]
            else:
                mod = [[], ]
                for _i in range(1, _num + 1):
                    selected_site = [_ for _ in combinations(possible_site, _i)]
                    mod.extend([[(one_site, _standard_mod) for one_site in each_site_comb] for each_site_comb in selected_site])
            if mods:
                new_mod = [_ + __ for _ in mod for __ in mods]
                mods.extend(new_mod)
            else:
                mods = mod
        if mods:
            mod_pep = [mod_processor.add_mod(pep, one_mod) for one_mod in mods]
        else:
            mod_pep = [pep]
    else:
        mod_pep = [pep]
    return mod_pep


def add_target_charge(modpep, charge=(2, 3)):
    """
    :param modpep: the list of modified peps
    :param charge: a tuple or an integer of targeted charge state
    """
    if isinstance(charge, int):
        charge = (charge, )
    prec_list = []
    for c in charge:
        _prec = rapid_kit.assemble_prec(modpep, c)
        prec_list.append(_prec)
    return prec_list
