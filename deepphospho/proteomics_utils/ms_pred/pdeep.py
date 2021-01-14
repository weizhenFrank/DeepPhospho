from ._pdeep_constant import BasicpDeepInfo
from ._pdeep_constant import MOD

import re
import os
from collections import defaultdict
import pandas as pd

from deepphospho.proteomics_utils import rapid_kit
from deepphospho.proteomics_utils.post_analysis.spectronaut import SpectronautLibrary


def intprec_to_pdeep_test(intprec_list):
    """
    从 intprec 转换为 pDeep2 的 test input
    intprec: 如 DKEAIQA4SESLMTSAPK.2
    pDeep2 test input 格式为
        peptide modification    charge
        FRTPSFLK    3,Phospho[T];5,Phospho[S];  2
        ...
    """
    title = ['peptide', 'modification',	'charge']
    pdeep_test_data_list = []
    for each_intprec in intprec_list:
        intprec_result = intprec_to_pdeep(each_intprec)
        if intprec_result is not None:
            stripped_pep, mod_info, charge = intprec_result
        else:
            continue
        pdeep_test_data_list.append([stripped_pep, mod_info, charge])
    pdeep_test_df = pd.DataFrame(pdeep_test_data_list, columns=title)
    return pdeep_test_df


def intprec_to_pdeep(intprec: str):
    int_to_pdeep2_mod = {
        'C': 'Carbamidomethyl[C]',
        '1': 'Oxidation[M]',
        '2': 'Phospho[S]',
        '3': 'Phospho[T]',
        '4': 'Phospho[Y]',
    }
    intseq, charge = intprec.split('.')
    if intseq.startswith('@'):
        intseq = intseq[1:]
    elif intseq.startswith('*'):
        return None
    else:
        pass
    stripped_pep = intseq.replace('1', 'M').replace('2', 'S').replace('3', 'T').replace('4', 'Y')
    mod_info = ''
    for _ in re.finditer('[C1234]', intseq):
        site = _.end()
        mod_char = _.group()
        mod = int_to_pdeep2_mod[mod_char]
        mod_info += f'{site},{mod};'
    return stripped_pep, mod_info, charge


def mod_extraction_for_pdeep(mod_pep):
    """
    The
    """
    mod_pep = mod_pep.replace('_', '')
    if '[' not in mod_pep:
        return ''
    else:
        modinfo = ''
        mod_start = [left_bracket.start() for left_bracket in re.finditer('\[', mod_pep)]
        mod_end = [right_bracket.start() for right_bracket in re.finditer(']', mod_pep)]
        mod_len = 0
        for mod_site in zip(mod_start, mod_end):
            if mod_site[0] == 0:  # or mod_site[1] == len(mod_pep) - 1:
                return 'Unsupport'
            else:
                mod_residu = mod_pep[mod_site[0] - 1]
                mod_type = mod_pep[mod_site[0] + 1: mod_site[1]].replace(' ', '')
                mod_type = re.sub(r'\(.+?\)', f'[{mod_residu}]', mod_type)
                modinfo += '{mod_site},{mod_type};'.format(mod_site=mod_site[0] - mod_len, mod_type=mod_type)
                mod_len += (mod_site[1] - mod_site[0] + 1)
        return modinfo


def inten_dict_to_plabel(inten_dict: dict):
    """
    :param inten_dict: The input dict should have the k[v] pairs as 'Prec': {'Frag_1': Inten_1, 'Frag_2': Inten_2, ...}
    """
    plabel_rows = []
    for prec, ion_inten_dict in inten_dict.items():
        intprec_trans = intprec_to_pdeep(prec)
        if intprec_trans is None:
            continue
        stripped_pep, mod_info, charge = intprec_trans
        spec = f'Unknown.{charge}.0.0'
        plabel_ion_str = plabel_one_ion_row(ion_inten_dict, return_type='str')
        plabel_rows.append(f'{spec}\t{stripped_pep}\t{mod_info}\t{plabel_ion_str}')
    return plabel_rows


def write_plabel_with_inten_dict(inten_dict: dict, output_path: str):
    plabel_rows = inten_dict_to_plabel(inten_dict)
    with open(output_path, 'w') as f:
        f.write('spec\tpeptide\tmodinfo\tb\tb-NH3\tb-H2O\tb-ModLoss\ty\ty-NH3\ty-H2O\ty-ModLoss\n')
        for row in plabel_rows:
            f.write(row + '\n')


def plabel_to_pred_input(plabel_path):
    plabel_df = pd.read_csv(plabel_path, sep='\t', low_memory=False)
    plabel_df['charge'] = plabel_df['spec'].apply(lambda x: x.split('.')[-3])
    plabel_df = plabel_df[['peptide', 'modinfo', 'charge']]
    plabel_df.columns = ['peptide', 'modification', 'charge']
    return plabel_df


def plabel_one_ion_row(ion_inten_dict: dict,
                       ion_type=('b', 'b-NH3', 'b-H2O', 'b-ModLoss', 'y', 'y-NH3', 'y-H2O', 'y-ModLoss'),
                       return_type='str'):
    ion_dict = defaultdict(list)
    ion_dict.fromkeys(ion_type)
    loss_trans = {'1,H3PO4': 'ModLoss',
                  '1,H2O': 'H2O',
                  '1,NH3': 'NH3'}
    for frag, inten in ion_inten_dict.items():
        frag_type, frag_num, frag_charge, frag_loss = re.findall(r'([abcxyz])(\d+)\+(\d)-(.+)', frag)[0]
        if frag_loss == 'Noloss':
            ion_name = f'{frag_type}'
        elif frag_loss in ['1,H2O', '1,NH3', '1,H3PO4']:
            ion_name = f'{frag_type}-{loss_trans[frag_loss]}'
        else:
            continue
        ion_dict[ion_name].append((f'{frag_type}{frag_num}{ion_name[1:]}+{frag_charge},{inten};',
                                   int(frag_num),
                                   int(frag_charge)))
    if return_type == 'dict':
        return ion_dict
    elif return_type == 'str':
        ion_info = []
        for each_ion_type in ion_type:
            if each_ion_type[0] in ['a', 'b', 'c']:
                sorted_ions = sorted(ion_dict[each_ion_type], key=lambda x: (x[2], x[1]), reverse=False)
            elif each_ion_type[0] in ['x', 'y', 'z']:
                sorted_ions = sorted(ion_dict[each_ion_type], key=lambda x: (-x[2], x[1]), reverse=True)
            else:
                raise
            ions = [_[0] for _ in sorted_ions]
            ion_info.append(''.join(ions))
        return '\t'.join(ion_info)


def plabel_ion_info(one_psm_df, return_type):
    ion_info = {'b': '', 'b-NH3': '', 'b-H2O': '', 'b-ModLoss': '', 'y': [], 'y-NH3': [], 'y-H2O': [], 'y-ModLoss': []}
    for row_index, each_row in one_psm_df.iterrows():
        fragment_type = each_row['FragmentType']
        fragment_num = each_row['FragmentNumber']
        fragment_charge = each_row['FragmentCharge']
        fragment_relative_intensity = each_row['RelativeIntensity']
        fragment_losstype = each_row['FragmentLossType']
        if fragment_type == 'b':
            if fragment_losstype == 'noloss':
                ion_info['b'] += 'b{num}+{charge},{relative_intensity};'.format(num=fragment_num, charge=fragment_charge, relative_intensity=fragment_relative_intensity)
            elif fragment_losstype == 'NH3':
                ion_info['b-NH3'] += 'b{num}-NH3+{charge},{relative_intensity};'.format(num=fragment_num, charge=fragment_charge, relative_intensity=fragment_relative_intensity)
            elif fragment_losstype == 'H2O':
                ion_info['b-H2O'] += 'b{num}-H2O+{charge},{relative_intensity};'.format(num=fragment_num, charge=fragment_charge, relative_intensity=fragment_relative_intensity)
            elif fragment_losstype == 'H3PO4':
                ion_info['b-ModLoss'] += 'b{num}-ModLoss+{charge},{relative_intensity};'.format(num=fragment_num, charge=fragment_charge, relative_intensity=fragment_relative_intensity)
            else:
                continue
        elif fragment_type == 'y':
            if fragment_losstype == 'noloss':
                ion_info['y'].append('y{num}+{charge},{relative_intensity};'.format(num=fragment_num, charge=fragment_charge, relative_intensity=fragment_relative_intensity))
            elif fragment_losstype == 'NH3':
                ion_info['y-NH3'].append('y{num}-NH3+{charge},{relative_intensity};'.format(num=fragment_num, charge=fragment_charge, relative_intensity=fragment_relative_intensity))
            elif fragment_losstype == 'H2O':
                ion_info['y-H2O'].append('y{num}-H2O+{charge},{relative_intensity};'.format(num=fragment_num, charge=fragment_charge, relative_intensity=fragment_relative_intensity))
            elif fragment_losstype == 'H3PO4':
                ion_info['y-ModLoss'].append('y{num}-ModLoss+{charge},{relative_intensity};'.format(num=fragment_num, charge=fragment_charge, relative_intensity=fragment_relative_intensity))
            else:
                continue
    if return_type == 'dict':
        return ion_info
    elif return_type == 'str':
        str_ion_info = ''
        b_ion_order = ['b', 'b-NH3', 'b-H2O', 'b-ModLoss']
        #   ion_order = ['b', 'y']
        for ion_losstype in b_ion_order:
            str_ion_info += ion_info[ion_losstype]
            str_ion_info += '\t'

        y_ion_order = ['y', 'y-NH3', 'y-H2O', 'y-ModLoss']
        for ion_losstype in y_ion_order:
            str_ion_info += ''.join(ion_info[ion_losstype][::-1])
            if ion_losstype != 'y-ModLoss':
                str_ion_info += '\t'
            #   if ion_losstype != 'y':
            #       str_ion_info += '\t'

        return str_ion_info


def sn_lib_to_plabel(lib, plabel_output):
    if isinstance(lib, pd.DataFrame):
        lib_df = lib
    else:
        if os.path.exists(lib):
            lib_df = pd.read_csv(lib, sep='\t', low_memory=False)
        else:
            raise FileNotFoundError
    lib_df['Prec'] = lib_df['ModifiedPeptide'] + '.' + lib_df['PrecursorCharge'].astype(str)
    with open(plabel_output, 'w') as plabel_handle:
        plabel_handle.write('spec\tpeptide\tmodinfo\tb\tb-NH3\tb-H2O\tb-ModLoss\ty\ty-NH3\ty-H2O\ty-ModLoss\n')
        #   handle_plabel.write('spec\tpeptide\tmodinfo\tb\ty\n')
        for psm_index, (each_prec, each_psm_df) in enumerate(lib_df.groupby('Prec')):
            first_row = each_psm_df.iloc[0]
            spec = '{title}.{charge}.0.0'.format(title=first_row['ReferenceRun'], charge=first_row['PrecursorCharge'])
            #   spec = '{charge}.0.0'.format(charge=first_fragment[1])
            stripped_pep = first_row['StrippedPeptide']
            mod_pep = first_row['ModifiedPeptide']
            modinfo = mod_extraction_for_pdeep(mod_pep)
            if modinfo == 'Unsupport':
                continue
            ion_info = plabel_ion_info(each_psm_df, 'str')
            plabel_handle.write('{spec}\t{pep}\t{mod}\t{ioninfo}\n'.format(
                spec=spec, pep=stripped_pep, mod=modinfo, ioninfo=ion_info))


def sn_lib_to_pdeep_test(test_lib, test_set_output):
    if isinstance(test_lib, pd.DataFrame):
        lib_df = test_lib
    else:
        if os.path.exists(test_lib):
            lib_df = pd.read_csv(test_lib, sep='\t', low_memory=False)
        else:
            raise FileNotFoundError
    lib_df['Prec'] = lib_df['ModifiedPeptide'] + '.' + lib_df['PrecursorCharge'].astype(str)
    lib_df = lib_df.drop_duplicates('Prec')
    with open(test_set_output, 'w') as test_handle:
        test_handle.write('peptide\tmodification\tcharge\n')

        for row_index, each_lib_row in lib_df.iterrows():
            mod_pep = each_lib_row['ModifiedPeptide']
            charge = str(each_lib_row['PrecursorCharge'])
            stripped_pep = each_lib_row['StrippedPeptide']
            mod = mod_extraction_for_pdeep(mod_pep)
            if mod == 'Unsupport':
                continue
            test_handle.write('{}\t{}\t{}\n'.format(stripped_pep, mod, charge))


def extract_pdeep_mod(mod_pep, mod_ident='bracket', mod_trans=True):
    """
    input: '_C[Carbamidomethyl (C)]DM[Oxidation (M)]EDER_'
    output: 'CDMEDER', '1,Carbamidomethyl[C];3,Oxidation[M];'
    """
    stripped_pep, mod = rapid_kit.split_mod(modpep=mod_pep, mod_ident=mod_ident)
    if mod_trans:
        mod = trans_sn_mod(mod)
    return stripped_pep, mod


def trans_sn_mod(mod):
    for sn_mod, pdeep_mod in MOD.items():
        mod = mod.replace(sn_mod, pdeep_mod)
        if '(' not in mod:
            break
    if '(' in mod:
        return None
    return mod


def restore_pdeep_mod_site(stripped_pep, mod_content, mod_processor):
    """
    This will restore the modification to stripped peptide.
    EXAMPLE: restore_pdeep_mod_site('MPALAIMGLSLAAFLELGMGASLCLSQQFK', '24,Carbamidomethyl[C];')
    -> 'MPALAIMGLSLAAFLELGMGASLC[Carbamidomethyl (C)]LSQQFK'
    """
    return rapid_kit.add_mod(stripped_pep, mod_content, mod_processor)


def pdeep_input(output_path, prec_list):
    with open(output_path, 'w') as out_file:
        pred_title = ['peptide', 'modification', 'charge']
        out_file.write('\t'.join(pred_title) + '\n')
        for _prec in prec_list:
            modpep, charge = rapid_kit.split_prec(_prec)
            strip_pep, mod = extract_pdeep_mod(modpep)
            out_file.write(f'{strip_pep}\t{mod}\t{charge}\n')


def pdeep_trainset(output_path, prec_inten_dict):
    with open(output_path, 'w') as out_file:
        plabel_title_list = BasicpDeepInfo.pDeepTrainsetTitle
        plabel_title = '\t'.join(plabel_title_list)
        out_file.write(plabel_title + '\n')
        for _prec, inten_dict in prec_inten_dict.items():
            plabel_row_dict = plabel_one_row_dict(_prec, inten_dict)
            if not plabel_row_dict:
                continue
            one_row_list = [plabel_row_dict[_] for _ in plabel_title_list]
            out_file.write('\t'.join(one_row_list) + '\n')


def plabel_one_row_dict(prec, inten_dict: dict):
    plabel_row_dict = defaultdict(str)
    modpep, charge = rapid_kit.split_prec(prec)
    strip_pep, mod = extract_pdeep_mod(modpep, mod_ident='bracket', mod_trans=True)
    if not mod:
        return None
    plabel_row_dict['spec'] = f'{charge}.0.0'
    plabel_row_dict['peptide'] = strip_pep
    plabel_row_dict['modinfo'] = mod
    for frag, inten in inten_dict.items():
        frag_type, frag_num, frag_charge, frag_loss = rapid_kit.split_fragment_name(frag)
        if frag_loss == 'noloss':
            plabel_type = frag_type
            plabel_frag = f'{frag_type}{frag_num}+{frag_charge}'
        elif frag_loss == 'NH3' or frag_loss == 'H2O':
            plabel_type = f'{frag_type}-{frag_loss}'
            plabel_frag = f'{frag_type}{frag_num}-{frag_loss}+{frag_charge}'
        else:
            plabel_type = f'{frag_type}-ModLoss'
            plabel_frag = f'{frag_type}{frag_num}-ModLoss+{frag_charge}'
        plabel_row_dict[plabel_type] += f'{plabel_frag},{inten};'
    return plabel_row_dict


def read_pdeep_result(pdeep_result, modloss_name='H3PO4',
                      require_mz=True, min_inten_ratio=0.01, min_frag_num=3,
                      exclude_frag_num=(1, 2), exclude_modloss=False):
    mod_dict = {'Carbamidomethyl[C]': '[Carbamidomethyl (C)]',
                'Oxidation[M]': '[Oxidation (M)]',
                'Phospho[S]': '[Phospho (STY)]',
                'Phospho[T]': '[Phospho (STY)]',
                'Phospho[Y]': '[Phospho (STY)]',
                }
    with open(os.path.abspath(pdeep_result), 'r') as pdeep_handle:
        predicted_fragment_data = dict()
        for each_line in pdeep_handle:
            each_line = each_line.strip('\n')
            if each_line == 'BEGIN IONS':
                fragment_dict = dict()
            elif each_line == 'END IONS':
                if len(fragment_dict) >= min_frag_num:
                    predicted_fragment_data[prec] = fragment_dict
                else:
                    pass
            else:
                if each_line.startswith('TITLE'):
                    split_pep_title = each_line.replace('TITLE=', '').split('|')
                    stripped_pep = split_pep_title[0]
                    mod = split_pep_title[1].strip(';')
                    charge = split_pep_title[2]

                    if not mod:
                        prec = '_{}_.{}'.format(stripped_pep, charge)
                    else:
                        mod_pep = ''
                        previous_mod_site = 0
                        for each_mod in mod.split(';'):
                            each_mod_info = each_mod.split(',')
                            mod_site = int(each_mod_info[0])
                            mod_type = mod_dict[each_mod_info[1]]
                            mod_pep += stripped_pep[previous_mod_site: mod_site] + mod_type
                            previous_mod_site = mod_site
                        mod_pep += stripped_pep[previous_mod_site:]
                        prec = '_{}_.{}'.format(mod_pep, charge)
                elif each_line[0].isdigit():
                    split_frag_inten_line = each_line.split(' ')
                    frag_inten = round(float(split_frag_inten_line[1]), 5) * 100
                    if frag_inten < min_inten_ratio:
                        continue
                    frag_mz = split_frag_inten_line[0]
                    if float(frag_mz) < 10:
                        continue
                    frag_name = split_frag_inten_line[2]

                    frag_type, frag_num, loss_type, frag_c = re.findall('([by])(\d+)-?(.+)?\+(\d)', frag_name)[0]
                    if int(frag_num) in exclude_frag_num:
                        continue
                    if exclude_modloss and loss_type == 'ModLoss':
                        continue
                    new_frag_name = f'{frag_type}{frag_num}+{frag_c}'
                    if not loss_type:
                        new_frag_name += '-noloss'
                    else:
                        new_frag_name += f'-{loss_type}' if loss_type != 'ModLoss' else f'-{modloss_name}'
                    if require_mz:
                        fragment_dict[new_frag_name] = (frag_mz, frag_inten)
                    else:
                        fragment_dict[new_frag_name] = frag_inten
                else:
                    continue
    return predicted_fragment_data


def trans_pdeep2_result_to_df(result: dict, frag_trans=None, pep_trans=None, pep_trans_col='IntPep') -> pd.DataFrame:
    df_rows = []
    for prec, inten_dict in result.items():
        if frag_trans is not None:
            inten_dict = {frag_trans[frag]: inten for frag, inten in inten_dict.items()}
        one_row = [prec, inten_dict]
        if pep_trans is not None:
            modpep, charge = prec.split('.')
            transed_pep = pep_trans(modpep)
            one_row.append(transed_pep)


def read_inten_from_plabel(_plabel_file):
    ion_type_list = ['b', 'b-NH3', 'b-H2O', 'b-ModLoss', 'y', 'y-NH3', 'y-H2O', 'y-ModLoss']
    _p_df = pd.read_csv(_plabel_file, sep='\t')
    _p_df = _p_df.fillna('')
    _p_df['prec'] = _p_df.apply(lambda x: '|'.join([x['peptide'], x['modinfo'], x['spec'].split('.')[-3]]), axis=1)

    _p_inten_dict = dict()

    def _merge_plabel_inten(x):
        _one_prec = x['prec']
        _one_inten_info = ''.join(x[ion_type_list].tolist()).split(';')[:-1]
        _p_inten_dict[_one_prec] = dict([(_o_f.split(',')[0], float(_o_f.split(',')[1])) for _o_f in _one_inten_info])

    _p_df.progress_apply(_merge_plabel_inten, axis=1)
    return _p_inten_dict


class pDeepSpectronaut(SpectronautLibrary):
    def __init__(self, spectronaut_version=12):
        super(pDeepSpectronaut, self).__init__(spectronaut_version)
        self.plabel_title_list = BasicpDeepInfo.pDeepTrainsetTitle

    def prec_ion_info(self, one_psm_df: pd.DataFrame, spectronaut_run_name=True):
        """
        For pDeep trainset preparation.
        This will receive get_one_prefix_result dataframe of one psm block and assemble get_one_prefix_result pd.series as one row of the plabel dataframe.
        :param one_psm_df: This must contain columns after ['PrecursorCharge', 'StrippedPeptide', 'ModifiedPeptide',
        'FragmentType', 'FragmentNumber', 'FragmentCharge', 'RelativeIntensity', 'FragmentLossType']
        :param spectronaut_run_name: This can be choose as True or False and dont affect the result. This can make the plabel file have much information
        :return: A series as one plabel dataframe row
        """
        first_row = one_psm_df.iloc[0]
        prec_charge = first_row['PrecursorCharge']
        if spectronaut_run_name:
            run_title = first_row['ReferenceRun']
            spec = '{title}.{charge}.0.0'.format(title=run_title, charge=prec_charge)
        else:
            spec = '{charge}.0.0'.format(charge=prec_charge)

        stripped_pep = first_row['StrippedPeptide']
        mod_pep = first_row['ModifiedPeptide']
        stripped_pep, modinfo = extract_pdeep_mod(mod_pep)
        if modinfo == 'Unsupport':
            return 'Unsupport'
        current_prec_info = pd.Series(data=[spec, stripped_pep, modinfo] + [''] * 8, index=self.plabel_title_list)

        for row_index in one_psm_df.index:
            line_series = one_psm_df.loc[row_index]

            fragment_type = line_series['FragmentType']
            fragment_num = line_series['FragmentNumber']
            fragment_charge = line_series['FragmentCharge']
            fragment_relative_intensity = line_series['RelativeIntensity']
            fragment_losstype = line_series['FragmentLossType']
            if fragment_type == 'b':
                if fragment_losstype == 'noloss':
                    current_prec_info['b'] += 'b{num}+{charge},{relative_intensity};'.format(num=fragment_num, charge=fragment_charge, relative_intensity=fragment_relative_intensity)
                elif fragment_losstype == 'NH3':
                    current_prec_info['b-NH3'] += 'b{num}-NH3+{charge},{relative_intensity};'.format(num=fragment_num, charge=fragment_charge, relative_intensity=fragment_relative_intensity)
                elif fragment_losstype == 'H2O':
                    current_prec_info['b-H2O'] += 'b{num}-H2O+{charge},{relative_intensity};'.format(num=fragment_num, charge=fragment_charge, relative_intensity=fragment_relative_intensity)
                else:
                    current_prec_info['b-ModLoss'] += 'b{num}-ModLoss+{charge},{relative_intensity};'.format(num=fragment_num, charge=fragment_charge, relative_intensity=fragment_relative_intensity)
            elif fragment_type == 'y':
                if fragment_losstype == 'noloss':
                    current_prec_info['y'] += 'y{num}+{charge},{relative_intensity};'.format(num=fragment_num, charge=fragment_charge, relative_intensity=fragment_relative_intensity)
                elif fragment_losstype == 'NH3':
                    current_prec_info['y-NH3'] += 'y{num}-NH3+{charge},{relative_intensity};'.format(num=fragment_num, charge=fragment_charge, relative_intensity=fragment_relative_intensity)
                elif fragment_losstype == 'H2O':
                    current_prec_info['y-H2O'] += 'y{num}-H2O+{charge},{relative_intensity};'.format(num=fragment_num, charge=fragment_charge, relative_intensity=fragment_relative_intensity)
                else:
                    current_prec_info['y-ModLoss'] += 'y{num}-ModLoss+{charge},{relative_intensity};'.format(num=fragment_num, charge=fragment_charge, relative_intensity=fragment_relative_intensity)
        return current_prec_info

    def plabel_trainset(self, output_path, spectronaut_run_name=True):
        """
        Write get_one_prefix_result pDeep trainset file by calling function prec_ion_info to process the library dataframe
        """
        trainset_df = pd.DataFrame(columns=self.plabel_title_list)
        for each_psm_index in self.get_psm_block_index(self._lib_df):
            current_prec_info = self.prec_ion_info(self._lib_df.loc[each_psm_index[0]: each_psm_index[1]], spectronaut_run_name)
            if not isinstance(current_prec_info, pd.DataFrame):
                continue
            trainset_df = trainset_df.append(current_prec_info, ignore_index=True)
        trainset_df.to_csv(output_path, sep='\t', index=False)


def extract_bracket(str_with_bracket):
    bracket_start = [left_bracket.start() for left_bracket in re.finditer('\(', str_with_bracket)]
    bracket_end = [right_bracket.start() for right_bracket in re.finditer('\)', str_with_bracket)]
    return bracket_start, bracket_end


mod_dict = {'M(ox)': 'Oxidation[M]',
            'Y(ph)': "Phospho[Y]",
            'S(ph)': "Phospho[S]",
            'T(ph)': "Phospho[T]",
            }


def _plabel_from_mq(x):
    def pdeep_mod_extraction(mod_pep):
        mod_pep = mod_pep.replace('_', '')
        modinfo = ''
        mod_start, mod_end = extract_bracket(mod_pep)
        mod_len = 0
        for mod_site in zip(mod_start, mod_end):
            mod_type = mod_pep[mod_site[0] - 1: mod_site[1] + 1].replace(' ', '')
            mod_type = mod_dict[mod_type]
            modinfo += '{mod_site},{mod_type};'.format(mod_site=mod_site[0] - mod_len, mod_type=mod_type)
            mod_len += (mod_site[1] - mod_site[0] + 1)
        return modinfo

    ion_type_list = ['b', 'b-NH3', 'b-H2O', 'b-ModLoss', 'y', 'y-NH3', 'y-H2O', 'y-ModLoss']
    plabel_title = ['spec', 'peptide', 'modinfo', *ion_type_list]

    spec_name = '{}.{}.{}.{}.0.dta'.format(x['Raw file'], x['Scan number'], x['Scan number'], x['Charge'])
    pep = x['Sequence']
    mod_pep = x['Modified sequence']
    mod_info = pdeep_mod_extraction(mod_pep)

    ions = x['Matches']
    intens = x['Intensities']
    inten_dict = dict(zip(ion_type_list, [''] * 8))

    ion_intens_list = list(zip(ions.split(';'), intens.split(';')))
    b_ion_info = [_ for _ in ion_intens_list if _[0].startswith('b')]
    y_ion_info = [_ for _ in ion_intens_list if _[0].startswith('y')]

    for diff_ion_info in [b_ion_info, y_ion_info]:
        current_num = 0
        _mod_start = False
        _second_mod_start = False
        for ion, inten in diff_ion_info:

            if '*' in ion:
                if not _mod_start:
                    current_num = 0
                    _mod_start = True
            if '-' in ion:
                if _mod_start:
                    continue

            ion_type, ion_num = re.findall('([by])(\d+)', ion)[0]
            ion_num = int(ion_num)

            re_charge = re.findall('\((\d)\+\)', ion)
            if re_charge:
                ion_charge = re_charge[0]
            else:
                ion_charge = '1'

            if ion_num <= current_num and '*' in ion:
                _second_mod_start = True
                continue
            if '*' in ion and _second_mod_start:
                continue
            current_num = ion_num

            tag = ion_type
            if '*' in ion:
                tag += '-ModLoss'
            elif '-' in ion:
                tag += '-{}'.format(re.findall('-(.+)', ion)[0])

            inten_dict[tag] += '{}{}{}+{},{};'.format(ion_type,
                                                      ion_num,
                                                      '-' + tag.split('-')[1] if '-' in tag else '',
                                                      ion_charge,
                                                      inten
                                                      )

    one_psm_data = [spec_name, pep, mod_info, *[inten_dict[_] for _ in ion_type_list]]
    return one_psm_data


"""  NOTICE This one is for MQ > 1.6, in which the modifications added in the peptide sequence was set as Phospho (STY) but not (ph) in 1.5

def extract_bracket(str_with_bracket):
    bracket_start = [left_bracket.start() for left_bracket in re.finditer('\(', str_with_bracket)][::2]
    bracket_end = [right_bracket.start() for right_bracket in re.finditer('\)', str_with_bracket)][1::2]
    return bracket_start, bracket_end

mod_dict2 = {'M(Oxidation (M))': 'Oxidation[M]',
            'Y(Phospho (STY))' : "Phospho[Y]",
            'S(Phospho (STY))' : "Phospho[S]",
            'T(Phospho (STY))' : "Phospho[T]",}

def pdeep_mod_extraction(mod_pep):
    mod_pep = mod_pep.replace('_', '')
    modinfo = ''
    mod_start, mod_end = extract_bracket(mod_pep)
    mod_len = 0
    for mod_site in zip(mod_start, mod_end):
        mod_type = mod_pep[mod_site[0] - 1: mod_site[1] + 1]# .replace(' ', '')
        mod_type = mod_dict2[mod_type]
        modinfo += '{mod_site},{mod_type};'.format(mod_site=mod_site[0] - mod_len, mod_type=mod_type)
        mod_len += (mod_site[1] - mod_site[0] + 1)
    return modinfo
"""

