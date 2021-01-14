
import os
import re

import pandas as pd

from ._deeprt_constant import MOD
from mskit import rapid_kit


def newrt_format_to_deeprt(rt_df: pd.DataFrame, pep_col='IntPep'):
    rt_df = rt_df[rt_df[pep_col].apply(lambda x: True if x[0] != '*' else False)]
    rt_df['IntPep'] = rt_df['IntPep'].apply(lambda x: x[1:] if x[0] == '@' else x)
    return rt_df


def parse_deeprt_log(log_path):
    with open(log_path, 'r') as f:
        log_content = f.readlines()

    param_num = log_content[0].split(':')[1]
    epoch_num = (len(log_content) - 2) // 3
    log_info = []
    for epoch_idx in range(1, epoch_num + 1):
        rows = log_content[(epoch_idx - 1) * 3 + 1: epoch_idx * 3 + 1]
        train_loss = re.findall(r'Training Loss: (\d+?\.\d+) ', rows[0])[0]
        test_loss = re.findall(r'Testing Loss: (\d+?\.\d+) ', rows[1])[0]
        data_num, pearson, spearman = re.findall(r'Corr on (\d+) testing samples: (\d+?\.\d+) \| (\d+?\.\d+)', rows[2])[0]
        log_info.append((epoch_idx, train_loss, test_loss, pearson, spearman, data_num))
    log_df = pd.DataFrame(log_info, columns=['Epoch', 'TrainLoss', 'TestLoss', 'TestPearson', 'TestSpearman', 'DataNum'])
    log_df = log_df.astype({'Epoch': int, 'DataNum': int, 'TrainLoss': float, 'TestLoss': float, 'TestPearson': float, 'TestSpearman': float})
    return log_df, param_num


def deeprt_input(out_path, pep_list, mod_trans=True):
    with open(out_path, "w") as f:
        f.write("sequence\tRT\n")
        for pep in pep_list:
            if mod_trans:
                pep = trans_sn_mod(pep)
                if not pep:
                    continue
            f.write(pep + "\t" + "0.0000" + "\n")


def deeprt_trainset(output_path, modpep_rt_list, mod_trans=True):
    with open(output_path, 'w') as out_file:
        out_file.write('sequence\tRT\n')
        for each_modpep_rt in modpep_rt_list:
            each_modpep = each_modpep_rt[0]
            each_rt = each_modpep_rt[1]
            if mod_trans:
                each_modpep = trans_sn_mod(each_modpep)
                if not each_modpep:
                    continue
            out_file.write('{seq}\t{rt}\n'.format(seq=each_modpep, rt=each_rt))


def read_deeprt_pred(pred_file, method='pandas', mod_trans=True):
    if method == 'pandas':
        pred_output = pd.read_csv(os.path.abspath(pred_file), sep='\t')
        seq = pred_output['seq'].to_list()
        pred_rt = pred_output['predicted'].to_list()
    else:
        with open(pred_file, 'r') as _pred:
            pred_title = _pred.readline()
            seq = []
            pred_rt = []
            for each_line in _pred:
                split_line = each_line.strip('\n').split('\t')
                seq.append(split_line[0])
                pred_rt.append(split_line[2])
    if mod_trans:
        modpep_list = []
        for modpep in seq:
            for sn_mod, int_mod in MOD.items():
                modpep = modpep.replace(int_mod, sn_mod)
            modpep = f'_{modpep}_'
            modpep_list.append(modpep)
    else:
        modpep_list = seq
    return dict(zip(modpep_list, pred_rt))


def get_coincide_rtdata(rt_dict_1, rt_dict_2):
    modpep, rt_list_1, rt_list_2 = rapid_kit.get_coincide_data(rt_dict_1, rt_dict_2)
    return modpep, rt_list_1, rt_list_2


def trans_sn_mod(modpep):
    for sn_mod, int_mod in MOD.items():
        modpep = modpep.replace(sn_mod, int_mod)
        if '[' not in modpep:
            break
    if '[' in modpep:
        return None
    modpep = modpep.strip('_')
    return modpep
