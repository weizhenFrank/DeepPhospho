import copy
import os
import json
import random
import re
import typing

import numpy as np
import pandas as pd

from deep_phospho.proteomics_utils import rapid_kit as rk
from deep_phospho.proteomics_utils.post_analysis import spectronaut as SN
from deep_phospho.proteomics_utils.post_analysis import maxquant as MQ
from deep_phospho.proteomics_utils.modpep_format_trans import unimodpep_to_intseq


SEED = 666
random.seed(SEED)


def split_nparts(
        data: typing.Union[list, tuple, np.ndarray],
        ratios: typing.Union[str, list, tuple, np.ndarray],
        ratio_sep=',',
        assign_remainder='first_n',
        seed=SEED,
) -> list:
    if isinstance(ratios, str):
        ratios = [float(_) for _ in ratios.split(ratio_sep)]
    if len(ratios) == 1:
        return data

    n_data = len(data)
    if n_data < len(ratios):
        print()
        # or warning or raise
    ratios = np.asarray(ratios)
    split_n_data = (ratios / np.sum(ratios) * n_data).astype(int)
    if assign_remainder == 'first':
        split_n_data[0] += (n_data - np.sum(split_n_data))
    elif assign_remainder == 'last':
        split_n_data[-1] += (n_data - np.sum(split_n_data))
    elif assign_remainder == 'first_n':
        split_n_data[:(n_data - np.sum(split_n_data))] += 1
    elif assign_remainder == 'last_n':
        split_n_data[-(n_data - np.sum(split_n_data)):] += 1
    elif assign_remainder == 'no':
        pass
    else:
        raise ValueError('')

    cum_split_n_data = np.cumsum(split_n_data)

    if isinstance(seed, random.Random):
        r = seed
    else:
        r = random.Random(seed)

    shuffled_data = copy.deepcopy(data)
    r.shuffle(shuffled_data)

    split_data = [shuffled_data[:cum_split_n_data[0]]]
    for idx, n in enumerate(cum_split_n_data[1:], 1):
        split_data.append(shuffled_data[cum_split_n_data[idx - 1]: n])
    return split_data


def split_data_and_store(
        data_path: str,
        output_folder: str,
        inten_dict: dict,
        rt_df: pd.DataFrame,
        split_ratio:
        typing.Union[tuple, list]
) -> dict:
    data_name = os.path.splitext(os.path.basename(data_path))[0]

    split_prec = split_nparts(data=list(inten_dict.keys()), ratios=split_ratio)
    split_pep = split_nparts(data=rt_df['IntPep'].tolist(), ratios=split_ratio)

    result_path = dict()
    for idx in range(len(split_ratio)):
        data_type = ['Train', 'Val', 'Holdout'][idx]

        inten_data = {prec: i for prec, i in inten_dict.items() if prec in split_prec[idx]}
        _path = os.path.join(output_folder, f'{data_name}-Ion_{data_type}.json')
        result_path[f'Ion{data_type}'] = _path
        with open(_path, 'w') as f:
            json.dump(inten_data, f, indent=4)

        rt_data = rt_df[rt_df['IntPep'].isin(split_pep[idx])]
        _path = os.path.join(output_folder, f'{data_name}-RT_{data_type}.txt')
        result_path[f'RT{data_type}'] = _path
        rt_data.to_csv(_path, sep='\t', index=False)

    return result_path


def sn_lib_to_trainset(lib_path, output_folder, split_ratio=(0.8, 0.2)):
    """
    Only SN 13+ is supported
    """
    snlib = SN.SpectronautLibrary(lib_path)
    snlib.add_intpep()
    snlib.add_intprec()
    snlib.add_frag_name(trans_dict=SN.sn_constant.LossType.SN_to_Readable)

    lib_intens = snlib.get_frag_inten(prec_col='IntPrec')
    rt_df = snlib.get_rt_df(pep_col='IntPep')

    return split_data_and_store(
        data_path=lib_path, output_folder=output_folder,
        inten_dict=lib_intens, rt_df=rt_df,
        split_ratio=split_ratio
    )

# def sn_results_to_trainset(result_path, output_folder, split_ratio=(0.9, 0.1)):
#     """
#
#     """
#     data_name = os.path.splitext(os.path.basename(result_path))[0]
#
#     result = pd.read_csv(result_path, sep='\t')
#
#     result_cols = set(result.columns)
#     modpep_col = list(result_cols & {'EG.ModifiedPeptide', 'EG.ModifiedSequence'})[0]
#     result['IntPep'] = result[modpep_col].apply(SN.sn_modpep_to_intseq)
#     result['IntPrec'] = result['IntPep'] + '.' + result['FG.Charge'].astype(str)
#
#     if 'EG.Cscore' in result_cols:
#         one_prec_result = result.groupby('IntPrec').apply(lambda x: x.loc[x['EG.Cscore'].idxmax()]).copy()
#     elif 'EG.Qvalue' in result_cols:
#         one_prec_result = result.groupby('IntPrec').apply(lambda x: x.loc[x['EG.Qvalue'].idxmin()]).copy()
#     else:
#         one_prec_result = result.groupby('IntPrec').apply(lambda x: x.loc[x['EG.Qvalue'].idxmin()]).copy()
#
#     result['FragLossType'] = result['F.FrgLossType'].apply(lambda x: SN.sn_constant.LossType.SN_to_Readable[x])
#     result['Fragments'] = result['']
#
#     for prec, df in self._lib_df.groupby(prec_col):
#         lib_spec[prec] = dict(df[['FragName', 'RelativeIntensity']].values)


def mq_to_trainset(msms_path, output_folder, split_ratio=(0.8, 0.2), mq_version='1.5'):
    """
    mq_version can be 1.5 or 1.6
    """
    df = pd.read_csv(msms_path, sep='\t')

    df = df[pd.isna(df['Reverse'])].copy()
    df = df[df['Proteins'].apply(lambda x: False if pd.notna(x) and x.startswith('CON__') else True)].copy()

    if mq_version == '1.5':
        df['IntPep'] = df['Modified sequence'].apply(MQ.mq_modpep_to_intseq_1_5)
        phos_mod_name = '(ph)'
    elif mq_version == '1.6':
        df['IntPep'] = df['Modified sequence'].apply(MQ.mq_modpep_to_intseq_1_6)
        phos_mod_name = '(Phospho (STY))'
    else:
        raise
    df['IntPrec'] = df['IntPep'] + '.' + df['Charge'].astype(str)

    df = df[df.apply(
        rk.filter_prob,
        find_col='Modified sequence', prob_col='Phospho (STY) Probabilities',
        mod_name=phos_mod_name, recept_prob=0.75, refute_prob=0.75,
        axis=1
    )]

    try:
        df = df[pd.notna(df['Intensities'])].copy()
    except KeyError:
        df['Intensities'] = df['Intensity'].copy()
        df = df[pd.notna(df['Intensities'])].copy()

    max_score_df = df.groupby('IntPrec').apply(lambda x: x.loc[x['Score'].idxmax()]).copy()

    intens = max_score_df.apply(MQ.inten_from_mq, axis=1).to_dict()
    rt_df = df[['IntPep', 'Retention time']].groupby('IntPep').median().reset_index()
    rt_df.columns = ['IntPep', 'iRT']

    return split_data_and_store(
        data_path=msms_path, output_folder=output_folder,
        inten_dict=intens, rt_df=rt_df,
        split_ratio=split_ratio
    )


def easypqp_to_trainset(tsv_path, output_folder, split_ratio=(0.8, 0.2)):
    """
    """
    df = pd.read_csv(tsv_path, sep='\t')

    df['IntPep'] = df['ModifiedPeptideSequence'].apply(unimodpep_to_intseq)
    df['IntPrec'] = df['IntPep'] + '.' + df['PrecursorCharge'].astype(str)
    df = df[
        (~df['FragmentSeriesNumber'].isin((1, 2)))
        & (df['FragmentType'].isin(('b', 'y')))
        & (df['FragmentCharge'].isin((1, 2)))
        ].copy()

    allowed_loss_type = ['H3O4P1', 'H2O1', 'H3N1', '']
    loss_type_to_dp_format = {
        '': 'Noloss',
        'H2O1': '1,H2O',
        'H3N1': '1,NH3',
        'H3O4P1': '1,H3PO4',
    }

    def frag_anno_to_name(x):
        try:
            f_type, f_num, f_loss, f_c = re.findall('([by])(\\d+)-?(.+?)?\^(\\d)', x)[0]
        except ValueError and TypeError and IndexError:
            raise ValueError(f'Fragment annotation {x} in easypqp tsv library has no standard format like y5^2 or y5-H3O4P1^2')
        if f_loss not in allowed_loss_type:
            return np.nan
        return f'{f_type}{f_num}+{f_c}-{loss_type_to_dp_format[f_loss]}'

    df['FragName'] = df['Annotation'].apply(frag_anno_to_name)
    df = df[pd.notna(df['FragName'])].copy()

    df = df.groupby('IntPrec').filter(lambda x: len(x) >= 6).copy()

    lib_intens = dict()
    for prec, _df in df.groupby('IntPrec'):
        lib_intens[prec] = dict(df[['FragName', 'LibraryIntensity']].values)

    rt_df = df[['IntPep', 'NormalizedRetentionTime']].drop_duplicates('IntPep')
    rt_df.columns = ['IntPep', 'RT']

    return split_data_and_store(
        data_path=tsv_path, output_folder=output_folder,
        inten_dict=lib_intens, rt_df=rt_df,
        split_ratio=split_ratio
    )


def file_to_trainset(path, output_folder, file_type: str, split_ratio=(0.8, 0.2)) -> dict:
    if file_type.lower() == 'snlib':
        return sn_lib_to_trainset(path, output_folder, split_ratio)
    elif file_type.lower() == 'mq1.5':
        return mq_to_trainset(path, output_folder, split_ratio, '1.5')
    elif file_type.lower() == 'mq1.6':
        return mq_to_trainset(path, output_folder, split_ratio, '1.6')
    elif file_type.lower() == 'easypqp':
        return easypqp_to_trainset(path, output_folder, split_ratio)

    else:
        raise ValueError(f'Invalid train file type: {file_type}')
