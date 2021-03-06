import os

import pandas as pd

from deep_phospho.proteomics_utils import modpep_format_trans
from deep_phospho.proteomics_utils.post_analysis import spectronaut as SN
from deep_phospho.proteomics_utils.post_analysis import maxquant as MQ


def sn_lib_to_pred_input(lib_path, output_folder):
    data_name = os.path.splitext(os.path.basename(lib_path))[0]
    ion_pred_path = os.path.join(output_folder, f'{data_name}-Ion_PredInput.txt')
    rt_pred_path = os.path.join(output_folder, f'{data_name}-RT_PredInput.txt')

    snlib = SN.SpectronautLibrary(lib_path)
    snlib.add_intpep()
    snlib.add_intprec()
    lib = snlib.to_df()

    intprecs = lib['IntPrec'].drop_duplicates().tolist()
    with open(ion_pred_path, 'w') as f:
        f.write('sequence\n')
        for p in intprecs:
            f.write(p + '\n')

    lib['IntPep'].drop_duplicates().to_csv(rt_pred_path, index=False)
    return {
        'IonPred': ion_pred_path,
        'RTPred': rt_pred_path
    }


def sn_results_to_pred_input(result_path, output_folder):
    data_name = os.path.splitext(os.path.basename(result_path))[0]
    ion_pred_path = os.path.join(output_folder, f'{data_name}-Ion_PredInput.txt')
    rt_pred_path = os.path.join(output_folder, f'{data_name}-RT_PredInput.txt')

    result = pd.read_csv(result_path, sep='\t')
    result_cols = set(result.columns)
    modpep_col = list(result_cols & {'EG.ModifiedPeptide', 'EG.ModifiedSequence'})[0]
    result['IntPep'] = result[modpep_col].apply(SN.sn_modpep_to_intseq)
    result['IntPrec'] = result['IntPep'] + '.' + result['FG.Charge'].astype(str)

    intprecs = result['IntPrec'].drop_duplicates().tolist()
    with open(ion_pred_path, 'w') as f:
        f.write('sequence\n')
        for p in intprecs:
            f.write(p + '\n')

    result['IntPep'].drop_duplicates().to_csv(rt_pred_path, index=False)
    return {
        'IonPred': ion_pred_path,
        'RTPred': rt_pred_path
    }


def mq_to_pred_input(result_path, output_folder, mq_version='1.5'):
    """
    mq_version can be 1.5 or 1.6
    """
    data_name = os.path.splitext(os.path.basename(result_path))[0]
    ion_pred_path = os.path.join(output_folder, f'{data_name}-Ion_PredInput.txt')
    rt_pred_path = os.path.join(output_folder, f'{data_name}-RT_PredInput.txt')

    df = pd.read_csv(result_path, sep='\t')

    df = df[pd.isna(df['Reverse'])].copy()
    df = df[df['Proteins'].apply(lambda x: False if pd.notna(x) and x.startswith('CON__') else True)].copy()
    if 'Potential contaminant' in df.columns:
        df = df[pd.notna(df['Potential contaminant'])].copy()

    if mq_version == '1.5':
        df['IntPep'] = df['Modified sequence'].apply(MQ.mq_modpep_to_intseq_1_5)
    elif mq_version == '1.6':
        df['IntPep'] = df['Modified sequence'].apply(MQ.mq_modpep_to_intseq_1_6)
    else:
        raise
    df['IntPrec'] = df['IntPep'] + '.' + df['Charge'].astype(str)

    intprecs = df['IntPrec'].drop_duplicates().tolist()
    with open(ion_pred_path, 'w') as f:
        f.write('sequence\n')
        for p in intprecs:
            f.write(p + '\n')

    df['IntPep'].drop_duplicates().to_csv(rt_pred_path, index=False)
    return {
        'IonPred': ion_pred_path,
        'RTPred': rt_pred_path
    }


def pep_list_to_pred_input(pep_file, output_folder, pep_format):
    """
    :param pep_file: this file should have two columns for peptide ("peptide") and precursor charge ("charge") in tab-separated format
    :param output_folder
    :param pep_format: This should be set to "SN13", "MQ1.5", "MQ1.6", "Comet", or "DP"
    """
    data_name = os.path.splitext(os.path.basename(pep_file))[0]
    ion_pred_path = os.path.join(output_folder, f'{data_name}-Ion_PredInput.txt')
    rt_pred_path = os.path.join(output_folder, f'{data_name}-RT_PredInput.txt')

    df = pd.read_csv(pep_file, sep='\t')

    pep_format_trans_func = {
        'SN13': modpep_format_trans.sn13_to_intpep,
        'MQ1.5': modpep_format_trans.mq1_5_to_intpep,
        'MQ1.6': modpep_format_trans.mq1_6_to_intpep,
        'Comet': modpep_format_trans.comet_to_intpep,
        'DP': lambda x: x,
    }[pep_format]

    df['IntPep'] = df['peptide'].apply(pep_format_trans_func)
    df['IntPrec'] = df['IntPep'] + '.' + df['charge'].astype(str)

    intprecs = df['IntPrec'].drop_duplicates().tolist()
    with open(ion_pred_path, 'w') as f:
        f.write('sequence\n')
        for p in intprecs:
            f.write(p + '\n')

    df['IntPep'].drop_duplicates().to_csv(rt_pred_path, index=False)
    return {
        'IonPred': ion_pred_path,
        'RTPred': rt_pred_path
    }


def file_to_pred_input(path, output_folder, file_type: str):
    if file_type.lower() == 'snlib':
        return sn_lib_to_pred_input(path, output_folder)
    elif file_type.lower() == 'snresult':
        return sn_results_to_pred_input(path, output_folder)
    elif file_type.lower() == 'mq1.5':
        return mq_to_pred_input(path, output_folder, '1.5')
    elif file_type.lower() == 'mq1.6':
        return mq_to_pred_input(path, output_folder, '1.6')
    elif file_type.lower() == 'pepsn13':
        return pep_list_to_pred_input(path, output_folder, pep_format='SN13')
    elif file_type.lower() == 'pepmq1.5':
        return pep_list_to_pred_input(path, output_folder, pep_format='MQ1.5')
    elif file_type.lower() == 'pepmq1.6':
        return pep_list_to_pred_input(path, output_folder, pep_format='MQ1.6')
    elif file_type.lower() == 'pepcomet':
        return pep_list_to_pred_input(path, output_folder, pep_format='Comet')
    elif file_type.lower() == 'pepdp':
        return pep_list_to_pred_input(path, output_folder, pep_format='DP')
    else:
        raise ValueError(f'Invalid prediction file format: {file_type}')
