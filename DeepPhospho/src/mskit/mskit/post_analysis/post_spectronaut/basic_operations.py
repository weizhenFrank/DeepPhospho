from .sn_constant import SNLibraryTitle

import os
import pandas as pd
import numpy as np

from mskit import rapid_kit
from mskit import calc


def get_lib_prec(lib_df):
    prec = lib_df.apply(lambda x: x['ModifiedPeptide'] + '.' + str(x['PrecursorCharge']), axis=1)
    return prec


def select_target_run_df(df, region_ident, return_col_list=None):
    colname = 'R.Instrument (parsed from filename)'
    return rapid_kit.extract_df_with_col_ident(df, colname, region_ident, return_col_list)


def get_library_info(lib_path):
    lib_df = pd.read_csv(lib_path, sep='\t', low_memory=False)
    protein_groups_num = len(lib_df['ProteinGroups'].drop_duplicates())
    precursor_num = len((lib_df['PrecursorCharge'].astype(str) + lib_df['ModifiedPeptide']).drop_duplicates())
    modpep_num = len(lib_df['ModifiedPeptide'].drop_duplicates())
    return protein_groups_num, precursor_num, modpep_num


def merge_lib(main_lib, accomp_lib, drop_col=('ProteinGroups', )):
    """
    :param main_lib: main library
    :param accomp_lib: accompanying library. Overlapped precursors with main lib in this lib will be deleted
    :param drop_col:
    :return:
    """
    if not isinstance(main_lib, pd.DataFrame):
        main_lib = pd.read_csv(main_lib, sep='\t')
    if not isinstance(accomp_lib, pd.DataFrame):
        accomp_lib = pd.read_csv(accomp_lib, sep='\t')

    main_prec = set(main_lib.apply(lambda x: x['ModifiedPeptide'] + str(x['PrecursorCharge']), axis=1))
    accomp_lib = accomp_lib[~(accomp_lib['ModifiedPeptide'] + accomp_lib['PrecursorCharge'].astype(str)).isin(main_prec)]

    hybrid_lib = main_lib.append(accomp_lib)
    if drop_col:
        hybrid_lib = hybrid_lib.drop(drop_col, axis=1)
    return hybrid_lib


def norm_params(original_df, params_list, norm_func=np.nanmedian, focus_col='EG.ModifiedPeptide'):
    """
    This function is used to get the median value or others depends on the norm func of given params list when one focus is detected in multi reps
    """
    grouped_df = original_df.groupby(focus_col)
    norm_param_df = grouped_df[params_list].transform(norm_func)
    norm_colname = [_ + '_norm' for _ in params_list]
    norm_param_df.columns = norm_colname
    normed_df = pd.concat([original_df, norm_param_df], axis=1)
    return normed_df


def get_lib_rt_info(lib, pep_col='ModifiedPeptide', rt_col='iRT', return_type='dict'):
    rt_df = lib[[pep_col, rt_col]]
    non_redundant_rt_df = rt_df.drop_duplicates(pep_col)
    rt_dict = dict(non_redundant_rt_df.set_index(pep_col)[rt_col])
    if return_type == 'dict':
        return rt_dict
    elif return_type == 'list':
        return [(k, v) for k, v in rt_dict.items()]
    else:
        raise


def get_lib_fragment_info(lib, norm=False):
    """
    Extract fragment intensity info of precursors from library
    :param lib: spectral library path or dataframe
    :param norm:
    :return: A dict has precursors (modpep.charge) as key, and fragment info (a dict has key-value pair like 'b5+1-noloss': 21.1) as value
    """
    if not isinstance(lib, pd.DataFrame):
        if os.path.isfile(lib):
            lib = pd.read_csv(lib, sep='\t')
        else:
            raise
    if 'Precursor' not in lib.columns:
        lib['Precursor'] = lib.apply(lambda x: x['ModifiedPeptide'] + '.' + str(x['PrecursorCharge']), axis=1)

    def _get_group_frag_data(one_group):
        frags = one_group.apply(lambda x: x['FragmentType'] + str(x['FragmentNumber']) + '+' + str(x['FragmentCharge']) + '-' + x['FragmentLossType'], axis=1)
        intens = one_group['RelativeIntensity']
        if norm:
            calc.normalize_intensity(list(intens), norm)
        return dict(zip(frags, intens))

    frag_info = lib.groupby('Precursor').apply(_get_group_frag_data)
    frag_dict = dict(frag_info)
    return frag_dict


def write_lib(output_path, inten_dict: dict, rt_dict: dict, seq2protein_dict=None):
    """
    This function is based on the precursors in inten_dict.
    Seq not in rt_dict will pass
    Seq not in seq2protein_dict will make protein as ''
    Fragment with no loss type will be noloss
    :param output_path:
    :param inten_dict:
    :param rt_dict:
    :param seq2protein_dict:
    :return:
    """
    if seq2protein_dict:
        title_list = SNLibraryTitle.LibraryMainCol
    else:
        title_list = SNLibraryTitle.LibraryMainColPGOut

    with open(output_path, 'w') as _out:
        lib_title = '\t'.join(title_list)
        _out.write(lib_title + '\n')

        for each_prec, frag_dict in inten_dict.items():
            modpep, charge = rapid_kit.split_prec(each_prec, keep_underline=True)
            if modpep in rt_dict:
                irt = rt_dict[modpep]
            else:
                continue
            stripped_pep, mod = rapid_kit.split_mod(modpep, mod_ident='bracket')
            mod = mod.split(' ')[0]
            prec_mz = calc.calc_prec_mz(stripped_pep, charge, mod)
            if seq2protein_dict:
                protein_acc = seq2protein_dict[stripped_pep] if stripped_pep in seq2protein_dict else ''
                protein_acc = protein_acc if isinstance(protein_acc, str) else ';'.join(set(protein_acc))
            else:
                protein_acc = ''
            for frag_name, frag_inten in frag_dict.items():
                frag_type, frag_num, frag_charge, frag_loss = rapid_kit.split_fragment_name(frag_name)
                frag_loss = frag_loss if frag_loss else 'noloss'
                fragment_mz = calc.calc_fragment_mz(stripped_pep, frag_type, frag_num, frag_charge, mod=mod)
                one_row_list = [charge, modpep, stripped_pep, irt,
                                modpep, prec_mz, frag_loss, frag_num,
                                frag_type, frag_charge, fragment_mz, frag_inten]
                one_row_list = list(map(str, one_row_list))
                if seq2protein_dict:
                    one_row_list.append(protein_acc)
                lib_line = "\t".join(one_row_list)
                _out.write(lib_line + '\n')
