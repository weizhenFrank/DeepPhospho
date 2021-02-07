import shutil
import os
import numpy as np
import re
import argparse
from deep_phospho.model_utils.loss_func import PearsonLoss, SALoss, SA_Pearson_Loss, RMSELoss, \
    L1_SA_Pearson_Loss, MaskedLanguageLoss, SALoss_MSE, FocalLoss
from math import sqrt
import torch
from scipy.stats import pearsonr
from scipy.stats import spearmanr
from torch.nn import ModuleList
import copy


def get_clones(module, N):
    return ModuleList([copy.deepcopy(module) for i in range(N)])


def copy_files(path, dst):
    if os.path.isfile(path):
        shutil.copyfile(path, os.path.join(dst, os.path.basename(path)))
    elif os.path.isdir(path):
        shutil.copytree(path, os.path.join(dst, os.path.basename(path)))


def custom_sigmoid(tensor):
    exp = torch.exp(-tensor) + 1
    reciprocal = torch.reciprocal(exp)
    return reciprocal * 1.1


def match_frag(s: str, d: dict):
    prog = re.compile(s, flags=re.IGNORECASE)
    to_be_matched = list(d.keys())
    match_result = [prog.match(string).group() for string in to_be_matched if prog.match(string) is not None]
    match = len(match_result) > 0
    assert len(match_result) == 1 or len(match_result) == 0, 'Multiple Match!'
    # import ipdb

    if match:
        # ipdb.set_trace()
        return match, match_result[0]
    else:
        return match


def ion_types(aa_len, aa_index, configs):
    """
    the order of elements in ions matters
    """

    if configs['TRAINING_HYPER_PARAM']['pdeep2mode']:
        ions = [f'b{aa_index}\+1-noloss$', f'b{aa_index}\+2-noloss$',  # $ matters!
                f'b{aa_index}\+1-1,H3PO4$', f'b{aa_index}\+2-1,H3PO4$',
                f'y{aa_len - aa_index}\+1-noloss$', f'y{aa_len - aa_index}\+2-noloss$',
                f'y{aa_len - aa_index}\+1-1,H3PO4$', f'y{aa_len - aa_index}\+2-1,H3PO4$',
                ]
    elif configs['TRAINING_HYPER_PARAM']['only_two_ions']:
        ions = [f'b{aa_index}\+1-noloss$',
                f'y{aa_len - aa_index}\+1-noloss$'
                ]
    else:
        ions = [f'b{aa_index}\+1-noloss$', f'b{aa_index}\+2-noloss$', f'b{aa_index}\+1-1,H3PO4$',
                f'b{aa_index}\+1-2,H3PO4$', f'b{aa_index}\+1-1,H2O$', f'b{aa_index}\+1-1,NH3$',
                f'y{aa_len - aa_index}\+1-noloss$', f'y{aa_len - aa_index}\+2-noloss$',
                f'y{aa_len - aa_index}\+1-1,H3PO4$',
                f'y{aa_len - aa_index}\+1-2,H3PO4$', f'y{aa_len - aa_index}\+1-1,H2O$',
                f'y{aa_len - aa_index}\+1-1,NH3$',
                ]
    return ions


def intensity_load_check(configs, intensity: dict, loaded_data):
    """ To check whether the ion intensities loaded correctly

    :param intensity: Ion intensities (key is ion type, value is normalized intensity (0-1)
    :return:
    """
    if configs['TRAINING_HYPER_PARAM']['pdeep2mode']:
        ion_reg_types = [r'[by][0-9]+\+[12]-noloss$', r'[by][0-9]+\+[12]-1,H3PO4$', ]
    else:
        ion_reg_types = [r'[by][0-9]+\+[12]-noloss$', r'[by][0-9]+\+1-1,H2O$',
                         r'[by][0-9]+\+1-1,NH3$', r'[by][0-9]+\+1-[12],H3PO4$', ]
    matched = []
    for k, v in intensity.items():
        for ion_type in ion_reg_types:
            prog = re.compile(ion_type, flags=re.IGNORECASE)
            if prog.match(k) is not None:
                matched.append(k)
    assert len(matched) == len(loaded_data[loaded_data > 0]), 'Not loading completely!'


def get_index(string, *chars):
    loc = []
    for char in chars:
        if char in string:
            #     print(loc.append(test1.find("3")))
            loc.append(string.find(char))
            #     print(loc.append(test1.rfind("3")))
            loc.append(string.rfind(char))

    loc = set(loc)

    return loc


def get_pkl_path(path, configs):
    if configs['TRAINING_HYPER_PARAM']['only_two_ions']:
        pkl_path = path + 'only_two_ions' + '.pkl'
    else:
        if configs['TRAINING_HYPER_PARAM']['use_prosit_pretrain']:
            pkl_path = path + 'use_prosit_pretrain' + '.pkl'
        elif configs['TRAINING_HYPER_PARAM']['pdeep2mode']:
            if configs['TRAINING_HYPER_PARAM']['add_phos_principle']:
                pkl_path = path + 'use_pdeep2mode' + '-add_phos_principle' + '.pkl'
            else:
                pkl_path = path + 'use_pdeep2mode' + '-not_add_phos_principle' + '.pkl'
        else:
            if configs['TRAINING_HYPER_PARAM']['add_phos_principle']:
                pkl_path = path + 'not_use_pdeep2mode' + '-add_phos_principle' + '.pkl'
            else:
                pkl_path = path + 'not_use_pdeep2mode' + '-not_add_phos_principle' + '.pkl'
    return pkl_path


def get_loss_func(configs):
    if configs['TRAINING_HYPER_PARAM']['Bert_pretrain']:
        loss_func = MaskedLanguageLoss(only_acc_masked_token=configs['TRAINING_HYPER_PARAM']['accumulate_mask_only'])
    elif 'two_stage' in configs['TRAINING_HYPER_PARAM'] and configs['TRAINING_HYPER_PARAM']['two_stage']:
        loss_func = torch.nn.MSELoss()
        loss_func_cls = torch.nn.BCEWithLogitsLoss()
        return loss_func, loss_func_cls
    elif configs['TRAINING_HYPER_PARAM']['loss_func'] == "L1":
        loss_func = torch.nn.L1Loss()
    elif configs['TRAINING_HYPER_PARAM']['loss_func'] == "PearsonLoss":
        loss_func = PearsonLoss()
    elif configs['TRAINING_HYPER_PARAM']['loss_func'] == "SALoss":
        loss_func = SALoss()
    elif configs['TRAINING_HYPER_PARAM']['loss_func'] == "SA_Pearson_Loss":
        loss_func = SA_Pearson_Loss()
    elif configs['TRAINING_HYPER_PARAM']['loss_func'] == "L1_SA_Pearson_Loss":
        loss_func = L1_SA_Pearson_Loss()
    elif configs['TRAINING_HYPER_PARAM']['loss_func'] == "SALoss_MSE":
        loss_func = SALoss_MSE()
    elif configs['TRAINING_HYPER_PARAM']['loss_func'] == "MSE":
        loss_func = torch.nn.MSELoss()
    elif configs['TRAINING_HYPER_PARAM']['loss_func'] == "RMSE":
        loss_func = RMSELoss()
    elif configs['TRAINING_HYPER_PARAM']['loss_func'] == "FocalLoss":
        loss_func = FocalLoss()
    elif configs['TRAINING_HYPER_PARAM']['loss_func'] == "BCE":
        loss_func = torch.nn.BCEWithLogitsLoss()
    else:
        loss_func = None
        assert loss_func is not None

    return loss_func


def give_name_ion(len_pep: int, pred_mat):
    ions = {}
    for index in range(len_pep - 1):
        index += 1
        for ion_type in range(8):
            if pred_mat[index][ion_type] != -2 and pred_mat[index][ion_type] != -1 and pred_mat[index][ion_type] != 0 and pred_mat[index][ion_type] >= 0:

                if ion_type == 0:
                    key = f'b{index}+1-Noloss'
                elif ion_type == 1:
                    key = f'b{index}+2-Noloss'
                elif ion_type == 2:
                    key = f'b{index}+1-1,H3PO4'
                elif ion_type == 3:
                    key = f'b{index}+2-1,H3PO4'
                elif ion_type == 4:
                    key = f'y{len_pep - index}+1-Noloss'
                elif ion_type == 5:
                    key = f'y{len_pep - index}+2-Noloss'
                elif ion_type == 6:
                    key = f'y{len_pep - index}+1-1,H3PO4'
                else:
                    key = f'y{len_pep - index}+2-1,H3PO4'
                ions[key] = float(pred_mat[index][ion_type])
    return ions


def show_params_status(model):
    """
    Prints parameters of a model
    """
    st = {}
    strings = []
    total_params = 0
    for p_name, p in model.named_parameters():

        if not ('bias' in p_name.split('.')[-1] or 'bn' in p_name.split('.')[-1]):
            st[p_name] = ([str(x) for x in p.size()], np.prod(p.size()), p.requires_grad)
        total_params += np.prod(p.size())
    for p_name, (size, prod, p_req_grad) in st.items():
        strings.append("{:<60s}: {:<16s}({:8d}) ({})".format(
            p_name, '[{}]'.format(','.join(size)), prod, 'grad' if p_req_grad else '    '
        ))
    strings = '\n'.join(strings)
    return f"\n{strings}\n ----- \n \n{total_params / 1000000.0:.3f}M total parameters \n "


def get_parser(description):

    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--use_holdout', type=bool, default=False, help="whether to use holdout dataset")
    parser.add_argument('--exp_name', type=str, default='', help="expriments name for output dir")
    parser.add_argument('--GPU', type=str, default=None, help="index of GPU")
    parser.add_argument('--dataset', type=str, default=None, help="dataset name")

    parser.add_argument('--ad_hoc', default=None, help="ad_hoc operation")
    return parser.parse_args()


def RMSE(act, pred):
    '''
    accept two numpy arrays
    '''
    return sqrt(np.mean(np.square(act - pred)))


def Pearson(act, pred):
    return pearsonr(act, pred)[0]


def Spearman(act, pred):
    '''
    Note: there is no need to use spearman correlation for now
    '''
    return spearmanr(act, pred)[0]


def Delta_t95(act, pred):
    num95 = int(np.ceil(len(act) * 0.95))
    return 2 * sorted(abs(act - pred))[num95 - 1]


def Delta_tr95(act, pred):
    return Delta_t95(act, pred) / (max(act) - min(act))


def Delta_tn(act, pred, n):
    numn = int(np.ceil(len(act) * n))
    return 2 * sorted(abs(act - pred))[numn - 1]
