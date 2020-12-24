from ._deeprt_constant import MOD

import os
import pandas as pd

from mskit import rapid_kit


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
