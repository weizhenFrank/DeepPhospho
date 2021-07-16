import os
import json
import random

import pandas as pd

from deep_phospho.proteomics_utils import rapid_kit as rk
from deep_phospho.proteomics_utils.post_analysis import spectronaut as SN
from deep_phospho.proteomics_utils.post_analysis import maxquant as MQ


SEED = 666


def sn_lib_to_trainset(lib_path, output_folder, split_ratio=(0.9, 0.1)):
    """
    Only SN 13+ is supported
    """
    data_name = os.path.splitext(os.path.basename(lib_path))[0]

    ion_train_path = os.path.join(output_folder, f'{data_name}-Ion_Train.json')
    ion_val_path = os.path.join(output_folder, f'{data_name}-Ion_Val.json')
    rt_train_path = os.path.join(output_folder, f'{data_name}-RT_Train.txt')
    rt_val_path = os.path.join(output_folder, f'{data_name}-RT_Val.txt')

    snlib = SN.SpectronautLibrary(lib_path)
    snlib.add_intpep()
    snlib.add_intprec()
    snlib.add_frag_name(trans_dict=SN.sn_constant.LossType.SN_to_Readable)
    lib_intens = snlib.get_frag_inten(prec_col='IntPrec')

    ion_train_prec = random.sample(list(lib_intens.keys()), int(len(lib_intens) * split_ratio[0]))
    ion_train_intens = {}
    ion_val_intens = {}
    for p, i in lib_intens.items():
        if p in ion_train_prec:
            ion_train_intens[p] = i
        else:
            ion_val_intens[p] = i

    with open(ion_train_path, 'w') as f:
        json.dump(ion_train_intens, f, indent=4)

    with open(ion_val_path, 'w') as f:
        json.dump(ion_val_intens, f, indent=4)

    rt_df = snlib.get_rt_df(pep_col='IntPep')
    train_rt_df = rt_df.sample(frac=split_ratio[0], random_state=SEED)
    val_rt_df = rt_df[~rt_df['IntPep'].isin(train_rt_df['IntPep'].tolist())]
    train_rt_df.to_csv(rt_train_path, sep='\t', index=False)
    val_rt_df.to_csv(rt_val_path, sep='\t', index=False)

    return {
        'IonTrain': ion_train_path,
        'IonVal': ion_val_path,
        'RTTrain': rt_train_path,
        'RTVal': rt_val_path
    }


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


def mq_to_trainset(msms_path, output_folder, split_ratio=(0.9, 0.1), mq_version='1.5'):
    """
    mq_version can be 1.5 or 1.6
    """
    data_name = os.path.splitext(os.path.basename(msms_path))[0]

    ion_train_path = os.path.join(output_folder, f'{data_name}-Ion_Train.json')
    ion_val_path = os.path.join(output_folder, f'{data_name}-Ion_Val.json')
    rt_train_path = os.path.join(output_folder, f'{data_name}-RT_Train.txt')
    rt_val_path = os.path.join(output_folder, f'{data_name}-RT_Val.txt')

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
        df = df[pd.notna(df['Intensity'])].copy()

    max_score_df = df.groupby('IntPrec').apply(lambda x: x.loc[x['Score'].idxmax()]).copy()
    intens = max_score_df.apply(MQ.inten_from_mq, axis=1).to_dict()
    ion_train_prec = random.sample(list(intens.keys()), int(len(intens) * split_ratio[0]))
    ion_train_intens = {}
    ion_val_intens = {}
    for p, i in intens.items():
        if p in ion_train_prec:
            ion_train_intens[p] = i
        else:
            ion_val_intens[p] = i

    with open(ion_train_path, 'w') as f:
        json.dump(ion_train_intens, f, indent=4)

    with open(ion_val_path, 'w') as f:
        json.dump(ion_val_intens, f, indent=4)

    rt_df = df[['IntPep', 'Retention time']].groupby('IntPep').median().reset_index()
    rt_df.columns = ['IntPep', 'RT']
    train_rt_df = rt_df.sample(frac=split_ratio[0], random_state=SEED)
    val_rt_df = rt_df[~rt_df['IntPep'].isin(train_rt_df['IntPep'].tolist())]
    train_rt_df.to_csv(rt_train_path, sep='\t', index=False)
    val_rt_df.to_csv(rt_val_path, sep='\t', index=False)

    return {
        'IonTrain': ion_train_path,
        'IonVal': ion_val_path,
        'RTTrain': rt_train_path,
        'RTVal': rt_val_path
    }


def file_to_trainset(path, output_folder, file_type: str, split_ratio=(0.9, 0.1)) -> dict:
    if file_type.lower() == 'snlib':
        return sn_lib_to_trainset(path, output_folder, split_ratio)
    elif file_type.lower() == 'mq1.5':
        return mq_to_trainset(path, output_folder, split_ratio, '1.5')
    elif file_type.lower() == 'mq1.6':
        return mq_to_trainset(path, output_folder, split_ratio, '1.6')
    else:
        raise ValueError(f'Invalid train file type: {file_type}')
