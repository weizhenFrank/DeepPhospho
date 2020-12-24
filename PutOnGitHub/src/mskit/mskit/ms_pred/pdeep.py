from ._pdeep_constant import BasicpDeepInfo
from ._pdeep_constant import MOD

import re
import os
from collections import defaultdict
import pandas as pd

from mskit import rapid_kit
from mskit.post_analysis.post_spectronaut import SpectronautLibrary


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
    int_to_pdeep2_mod = {
        'C': 'Carbamidomethyl[C]',
        '1': 'Oxidation[M]',
        '2': 'Phospho[S]',
        '3': 'Phospho[T]',
        '4': 'Phospho[Y]',
    }
    pdeep_test_data_list = []
    for each_intprec in intprec_list:
        intseq, charge = each_intprec.split('.')
        stripped_pep = intseq.replace('1', 'M').replace('2', 'S').replace('3', 'T').replace('4', 'Y')
        mod_info = ''
        for _ in re.finditer('[C1234]', intseq):
            site = _.end()
            mod_char = _.group()
            mod = int_to_pdeep2_mod[mod_char]
            mod_info += f'{site},{mod};'
        pdeep_test_data_list.append([stripped_pep, mod_info, charge])
    pdeep_test_df = pd.DataFrame(pdeep_test_data_list, columns=title)
    return pdeep_test_df


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


def read_pdeep_result(pdeep_result, modloss_name='H3PO4'):
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
                predicted_fragment_data[prec] = fragment_dict
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
                    if frag_inten < 0.01:
                        continue
                    frag_mz = split_frag_inten_line[0]
                    frag_name = split_frag_inten_line[2]

                    frag_type, frag_num, loss_type, frag_c = re.findall('([by])(\d+)-?(.+)?\+(\d)', frag_name)[0]
                    if int(frag_num) <= 2:
                        continue
                    new_frag_name = f'{frag_type}{frag_num}+{frag_c}'
                    if not loss_type:
                        new_frag_name += '-noloss'
                    else:
                        new_frag_name += f'-{loss_type}' if loss_type != 'ModLoss' else f'-{modloss_name}'
                    fragment_dict[new_frag_name] = (frag_mz, frag_inten)
                else:
                    continue
    return predicted_fragment_data


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


def _plabel_from_mq(x):
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

