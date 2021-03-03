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
    snlib = SN.SpectronautLibrary(lib_path)
    snlib.add_intpep()
    snlib.add_intprec()
    snlib.add_frag_name(trans_dict=SN.sn_constant.LossType.SN_to_Readable)
    lib_intens = snlib.get_frag_inten(prec_col='IntPrec')

    ion_train_prec = random.sample(list(lib_intens.keys()), len(lib_intens) * split_ratio[0])
    ion_train_intens = {}
    ion_val_intens = {}
    for p, i in lib_intens.items():
        if p in ion_train_prec:
            ion_train_intens[p] = i
        else:
            ion_val_intens[p] = i

    with open(os.path.join(output_folder, 'Ion-TrainSet-FromSN.json'), 'w') as f:
        json.dump(ion_train_intens, f, indent=4)

    with open(os.path.join(output_folder, 'Ion-ValSet-FromSN.json'), 'w') as f:
        json.dump(ion_val_intens, f, indent=4)

    rt_df = snlib.get_rt_df(pep_col='IntPep')
    train_rt_df = rt_df.sample(frac=split_ratio[0], random_state=SEED)
    val_rt_df = rt_df[~rt_df['IntPep'].isin(train_rt_df['IntPep'].tolist())]
    train_rt_df.to_csv(os.path.join(output_folder, 'RT-TrainSet-FromSN.txt'), sep='\t', index=False)
    val_rt_df.to_csv(os.path.join(output_folder, 'RT-ValSet-FromSN.txt'), sep='\t', index=False)


def mq_to_trainset(msms_path, output_folder, split_ratio=(0.9, 0.1), mq_version='1.5'):
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

    max_score_df = df.groupby('IntPrec').apply(lambda x: x.loc[x['Score'].idxmax()]).copy()
    intens = max_score_df.apply(MQ.inten_from_mq, axis=1).to_dict()
    ion_train_prec = random.sample(list(intens.keys()), len(intens) * split_ratio[0])
    ion_train_intens = {}
    ion_val_intens = {}
    for p, i in intens.items():
        if p in ion_train_prec:
            ion_train_intens[p] = i
        else:
            ion_val_intens[p] = i

    with open(os.path.join(output_folder, 'Ion-TrainSet-FromMQ.json'), 'w') as f:
        json.dump(ion_train_intens, f, indent=4)

    with open(os.path.join(output_folder, 'Ion-ValSet-FromMQ.json'), 'w') as f:
        json.dump(ion_val_intens, f, indent=4)

    rt_df = df[['IntPep', 'Retention time']].groupby('IntPep').median().reset_index()
    rt_df.columns = ['IntPep', 'RT']
    train_rt_df = rt_df.sample(frac=split_ratio[0], random_state=SEED)
    val_rt_df = rt_df[~rt_df['IntPep'].isin(train_rt_df['IntPep'].tolist())]
    train_rt_df.to_csv(os.path.join(output_folder, 'RT-TrainSet-FromMQ.txt'), sep='\t', index=False)
    val_rt_df.to_csv(os.path.join(output_folder, 'RT-ValSet-FromMQ.txt'), sep='\t', index=False)
