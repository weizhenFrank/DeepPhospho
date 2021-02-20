import re

from deep_phospho.proteomics_utils.constants.mass import CompoundMass, Mass
from deep_phospho.proteomics_utils import rapid_kit


def calc_prec_mz(pep: str, charge: int = None, mod=None) -> float:
    """
    Example:
    pep = 'LGRPSLSSEVGVIICDISNPASLDEMAK'
    charge = 3
    mod = '15,Carbamidomethyl;26,Oxidation;'
    -> 992.1668502694666

    calc_prec_mz('_VQISPDS[Phospho (STY)]GGLPER_.2')
    -> 717.83486304735
    calc_prec_mz('_TPPRDLPT[Phospho (STY)]IPGVTSPSSDEPPM[Oxidation (M)]EAS[Phospho (STY)]QSHLRNSPEDK_.4')
    -> 1012.2026098813

    :param pep: peptide that is modified with num, e.g. ACDM1M, where 1 equals to M[Oxidation]
    :param charge: precursor charge
    :param mod: str like 1,Carbamidomethyl;3,Oxidation; or list like [(1, 'Carbamidomethyl'), (3, 'Oxidation')]
    :return: Precursor m/z

    TODO 直接给出多少个哪种修饰，或带位点的修饰，或肽段直接带有修饰（需要哪个参数说明）
    """
    if '.' in pep:
        pep, charge = rapid_kit.split_prec(pep)
    if '[' in pep:
        stripped_pep = re.sub(r'\[.+?\]', '', pep).replace('_', '')
        mod_in_pep = re.findall(r'\[(.+?) \(.+?\)\]', pep)
    else:
        stripped_pep = pep.replace('_', '')
        mod_in_pep = None

    pep_mass = 0.
    for aa in stripped_pep:
        pep_mass += Mass.ResMass[aa]
    pep_mass += CompoundMass.CompoundMass['H2O']
    pep_mass += Mass.ProtonMass * charge

    if mod:
        if isinstance(mod, str):
            mod = [_.split(',') for _ in mod.strip(';').split(';')]
        for _each_mod in list(zip(*mod))[1]:
            pep_mass += Mass.ModMass[_each_mod] if _each_mod != 'Carbamidomethyl' else 0.
    if mod_in_pep:
        for each_mod in mod_in_pep:
            if 'Carbamidomethyl' in each_mod:
                continue
            else:
                pep_mass += Mass.ModMass[each_mod]
    return pep_mass / charge


def calc_fragment_mz(pep, frag_type, frag_num, frag_charge, loss_type=None) -> float:
    """
    Example:
    calc_fragment_mz('_VIHDNFGIVEGLM[Oxidation (M)]TTVHAITAT[Phospho (STY)]QK_', 'y', 16, 1, '1,H3PO4')
    -> 1697.8890815221002

    calc_fragment_mz('_[Acetyl (Protein N-term)]SGSS[Phospho (STY)]SVAAMKK_', 'y', 8, 1, '1,H3PO4')
    -> 803.4443855000001

    calc_fragment_mz('_[Acetyl (Protein N-term)]SGSS[Phospho (STY)]SVAAMKK_', 'y', 10, 2, )
    -> 523.24102464735

    calc_fragment_mz('_[Acetyl (Protein N-term)]SGSS[Phospho (STY)]SVAAMKK_', 'b', 6, 1, '1,H3PO4')
    -> 529.2252678

    pep = 'LGRPSLSSEVGVIICDISNPASLDEMAK'
    frag_type = 'y'
    frag_num = 10
    frag_charge = 1
    mod = '15,Carbamidomethyl;26,Oxidation;'
    -> 1091.5037505084001

    :param pep: peptide sequence
    :param frag_type: support b and y ion
    :param frag_num:
    :param frag_charge:
    :param loss_type:
    :return: Fragment m/z
    """

    # TODO 重复使用已经得到的 pep 信息
    stripped_pep, mod_sites, mods = rapid_kit.substring_finder(pep.replace('_', ''))
    mods = [mod.strip('[]()').split(' ')[0] for mod in mods]
    mod_sites = list(map(int, mod_sites))
    frag_num = int(frag_num)
    frag_charge = int(frag_charge)
    frag_mass = Mass.ProtonMass * frag_charge
    if frag_type == 'b':
        mod_dict = dict(zip(mod_sites, mods))
        if 0 in mod_dict:
            frag_mass += Mass.ModMass[mod_dict[0]]

    elif frag_type == 'y':
        stripped_pep = stripped_pep[::-1]
        frag_mass += CompoundMass.CompoundMass['H2O']
        pep_len = len(stripped_pep)
        mod_sites = [pep_len - site + 1 for site in mod_sites]
        mod_dict = dict(zip(mod_sites, mods))
        if frag_num >= pep_len and (pep_len + 1) in mod_sites:
            frag_mass += Mass.ModMass[mod_dict[pep_len + 1]]
    else:
        raise NameError('Only b and y ion are supported')
    for i in range(frag_num):
        frag_mass += Mass.ResMass[stripped_pep[i]]
        if i + 1 in mod_dict:
            mod_type = mod_dict[i + 1]
            if 'Carbamidomethyl' in mod_type:
                continue
            frag_mass += Mass.ModMass[mod_type]

    if loss_type is None or loss_type.lower() == 'noloss':
        pass
    else:
        for loss_num, loss_compound in [loss.split(',') for loss in loss_type.split(';')]:
            frag_mass -= int(loss_num) * Mass.ModLossMass[loss_compound]

    return frag_mass / frag_charge


def calc_fragment_mz_old(pep, frag_type, frag_num, frag_charge, mod=None) -> float:
    """
    Example:
    pep = 'LGRPSLSSEVGVIICDISNPASLDEMAK'
    frag_type = 'y'
    frag_num = 10
    frag_charge = 1
    mod = '15,Carbamidomethyl;26,Oxidation;'
    -> 1091.5037505084001

    :param pep: peptide sequence
    :param frag_type: support b and y ion
    :param frag_num:
    :param frag_charge:
    :param mod:
    :return: Fragment m/z
    """
    frag_num = int(frag_num)
    frag_charge = int(frag_charge)
    frag_mass = Mass.ProtonMass * frag_charge

    if mod:
        if isinstance(mod, str):
            mod = [_.split(',') for _ in mod.strip(';').split(';')]
            mod = [(int(_[0]), _[1]) for _ in mod]
        mod_dict = dict(mod)
    else:
        mod_dict = dict()

    if frag_type == 'b':
        for i in range(frag_num):
            frag_mass += Mass.ResMass[pep[i]]
            if i + 1 in mod_dict:
                frag_mass += Mass.ModMass[mod_dict[i + 1]]
        if 0 in mod_dict:
            frag_mass += Mass.ModMass[mod_dict[0]]

    elif frag_type == 'y':
        frag_mass += CompoundMass.CompoundMass['H2O']
        pep_len = len(pep)
        for i in range(pep_len - 1, pep_len - 1 - frag_num, -1):
            frag_mass += Mass.ResMass[pep[i]]
            if i + 1 in mod_dict:
                frag_mass += Mass.ModMass[mod_dict[i + 1]]
        if frag_num == pep_len:
            frag_mass += Mass.ModMass[mod_dict[0]]
    else:
        raise NameError('Only b and y ion are supported')
    return frag_mass / frag_charge


def get_fragment_mz_dict(pep, fragments, mod=None):
    """
    :param pep:
    :param fragments:
    :param mod:
    :return:
    """
    mz_dict = dict()
    for each_fragment in fragments:
        frag_type, frag_num, frag_charge = rapid_kit.split_fragment_name(each_fragment)
        mz_dict[each_fragment] = calc_fragment_mz(
            pep, frag_type, frag_num, frag_charge, mod)
    return mz_dict
