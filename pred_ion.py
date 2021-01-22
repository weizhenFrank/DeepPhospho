import argparse
import copy
import datetime
import os
import random
import json
from tqdm import tqdm

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import ipdb

import torch
from torch.utils.data import DataLoader

from deep_phospho.models.ion_model import StackedLSTM  # Use the LSTMTransformer in EnsembleModel.py
from deep_phospho.models.EnsembelModel import LSTMTransformer

from deep_phospho.model_dataset.preprocess_input_data import IonData, Dictionary
from deep_phospho.model_dataset.dataset import IonDataset, collate_fn

from deep_phospho.model_utils.logger import setup_logger
from deep_phospho.model_utils.param_config_load import load_param_from_file
from deep_phospho.model_utils.ion_eval import SA, Pearson
from deep_phospho.model_utils.utils_functions import show_params_status
from deep_phospho.model_utils.utils_functions import give_name_ion


# ---------------- User defined space Start --------------------

# Define config path as the model work dir
ConfigPath = r''
WorkFolder = os.path.dirname(ConfigPath)

cfg = load_config(ConfigPath)

load_model_path = "/p300/projects/IonAndRT/result/ion_inten/AcData/ion_inten-PhosDIA_DIA18-LSTMTransformer-RemoveSigmoidRemove0AssignEpoch90OfJeffVeroE6R2P2-remove_ac_pepFalse-add_phos_principleTrue-LossTypeMSE-use_holdoutFalse-2020_10_29_09_37_41/ckpts/best_model.pth"

# ---------------- User defined space End --------------------
from deep_phospho.configs import ion_inten_config as cfg


def get_parser():
    parser = argparse.ArgumentParser(description='IonIntensity prediction Analysis')
    parser.add_argument('--exp_name', type=str, default='', help="expriments name for output dir")
    parser.add_argument('--GPU', default=None, help="index of GPU")
    parser.add_argument('--pretrain_param', default=None, type=str, help="path of pretrained_model")
    return parser.parse_args()


args = get_parser()

info = f'ion_inten-{cfg.Intensity_DATA_CFG["DataName"]}-{cfg.MODEL_CFG["model_name"]}-{args.exp_name}'
init_time = datetime.datetime.now().strftime("%Y%m%d_%H_%M_%S")
instance_name = f'{info}-{init_time}'

output_dir = os.path.join(WorkFolder, instance_name)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

logger = setup_logger("IonIntensity", output_dir)

SEED = 666
torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.backends.cudnn.deterministic = True
torch.autograd.set_detect_anomaly(True)

if cfg.TRAINING_HYPER_PARAM['GPU_INDEX']:
    device = torch.device(f'cuda:{cfg.TRAINING_HYPER_PARAM["GPU_INDEX"]}')
elif torch.cuda.is_available():
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')

dictionary = Dictionary()
idx2aa = dictionary.idx2word

Iontest = IonData(cfg.Intensity_DATA_CFG['HoldoutPATH'], dictionary=dictionary)
test_dataset = IonDataset(Iontest)
test_dataloader = DataLoader(dataset=test_dataset,
                             shuffle=False,
                             batch_size=64 * 8 * 3,
                             num_workers=0,
                             collate_fn=collate_fn)


def idxtoaa(arr):
    peptide = [idx2aa[int(aa_idx)] for aa_idx in arr]
    return ''.join(peptide).replace('#', '').replace('$', '')


if cfg.MODEL_CFG['model_name'] == "StackedLSTM":
    cfg_to_load = copy.deepcopy(cfg.MODEL_CFG)
    cfg_to_load.pop("pretrain_param")
    model = StackedLSTM(
        # ntoken=Ionholdout.N_aa,
        ntoken=31,
        # for prosit, it has 0-21
        # row_num=Ionholdout.row_num,
        row_num=53,
        # for prosit, it has 30 max length
        use_prosit=cfg.data_name == 'Prosit',
        **cfg_to_load,
    )
elif cfg.MODEL_CFG['model_name'] == "LSTMTransformer":
    cfg_to_load = copy.deepcopy(cfg.MODEL_CFG)
    model = LSTMTransformer(
        # ntoken=Iontrain.N_aa,
        RT_mode=False,
        ntoken=len(dictionary) - 1,  # before is 31
        # for prosit, it has 0-21
        use_prosit=cfg.Intensity_DATA_CFG["DataName"] == 'Prosit',
        pdeep2mode=cfg.TRAINING_HYPER_PARAM['pdeep2mode'],
        two_stage=cfg.TRAINING_HYPER_PARAM['two_stage'],
        **cfg_to_load,
    )
else:
    raise Exception("No model given!")

if args.pretrain_param is not None:
    load_model_path = args.pretrain_param

model = load_param_from_file(model,
                             load_model_path,
                             partially=False, logger_name='IonIntensity')

logger.info(str(model))
logger.info("model parameters statuts: \n%s" % show_params_status(model))

pred_matrix = []
y_matrix = []

pep_len = []
pep = []
charges = []

hidden_norm = []
pearson_eval = []
short_angle = []

if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    model = torch.nn.DataParallel(model)
model = model.cuda()
model.eval()
# ipdb.set_trace()
logger.info("Start Testing")
with torch.no_grad():
    if hasattr(model, "module"):
        if hasattr(model.module, "transformer_flag"):
            model.module.set_transformer()
    else:
        if hasattr(model, "transformer_flag"):
            if not model.transformer_flag:
                # ipdb.set_trace()
                model.set_transformer()
    logger.info("set transformer on")
    for idx, (inputs, y) in tqdm(enumerate(test_dataloader), total=len(test_dataloader)):

        # ipdb.set_trace()
        if len(inputs) == 3:
            seq_x = inputs[0]
            x_charge = inputs[1]
            x_gb = inputs[2]
            seq_x = seq_x.cuda()
            x_charge = x_charge.cuda()
            x_gb = x_gb.cuda()
            pred_y = model(x1=seq_x, x2=x_charge, x3=x_gb)
        else:
            seq_x = inputs[0]
            x_charge = inputs[1]
            seq_x = seq_x.cuda()
            x_charge = x_charge.cuda()
            # ipdb.set_trace()
            # try:
            #     pred_y = model(x1=seq_x, x2=x_charge)
            # except RuntimeError:
            #     ipdb.set_trace()
            pred_y = model(x1=seq_x, x2=x_charge)
        y = y.cuda()
        # ipdb.set_trace()

        if isinstance(pred_y, tuple):
            pred_y, hidden_vec_norm = pred_y

            # if hidden_vec_norm is not None:
            #     hidden_norm.append(hidden_vec_norm)

        pred_y[torch.where(y == -1)] = -1
        pred_y[torch.where(y == -2)] = -2

        pred_matrix.append(pred_y.detach().cpu())
        y_matrix.append(y.detach().cpu())

        pep.append(seq_x.detach().cpu())
        pep_charge = x_charge.reshape(x_charge.shape[0], -1)[:, 0]
        charges.append(pep_charge.detach().cpu())
        pep_len.append(((seq_x != 0).sum(axis=-1) - 2).detach().cpu())
        # torch.cuda.empty_cache()


pred_matrix_all = torch.cat(pred_matrix).numpy()
y_matrix_all = torch.cat(y_matrix).numpy()

pep = torch.cat(pep).numpy()
pep_len = torch.cat(pep_len).numpy()
charges = torch.cat(charges).numpy()

# if len(hidden_norm) != 0:
#     hidden_norm = torch.cat(hidden_norm)
# else:
#     hidden_norm = None


all_pred_ions_with_name = []
all_gt_ion_with_name = []
all_len = []
all_aa = []
all_charge = []

below_cut_counts = 0

# TODO This var controls whether with label or not, be True or False in Config (has been deleted)
No_Intensity = cfg.HOLDOUT_DATA_CFG['To_Predict']

for pred_inten_mat, pep_inten_mat, aas, length_aas, charge in zip(pred_matrix_all, y_matrix_all, pep, pep_len, charges):

    if not No_Intensity:
        # ipdb.set_trace()
        pred_inten__vec = pred_inten_mat.reshape(-1)
        pep_inten__vec = pep_inten_mat.reshape(-1)
        select = (pep_inten__vec != 0) * (pep_inten__vec != -1) * (pep_inten__vec != -2)
        if len(pep_inten__vec[select]) < 3:
            below_cut_counts += 1
            print("(pep_inten__vec != 0) * (pep_inten__vec != -1) * (pep_inten__vec != -2) < 3")
            continue

        pc = Pearson(pred_inten__vec[select], pep_inten__vec[select])
        pearson_eval.append(pc)
        sa = SA(pred_inten__vec[select], pep_inten__vec[select])
        short_angle.append(sa)
        all_gt_ion_with_name.append(give_name_ion(int(length_aas), pep_inten_mat))

    # ipdb.set_trace()
    all_pred_ions_with_name.append(give_name_ion(int(length_aas), pred_inten_mat))
    all_len.append(length_aas)
    all_aa.append(idxtoaa(aas))
    all_charge.append(charge)

if below_cut_counts > 0:
    logger.info(f"There is {below_cut_counts} precursors below cut off!")

logger.info("Start write into file")
if not No_Intensity:
    ion_pred = pd.DataFrame({"pred_with_name": all_pred_ions_with_name, "gt_ion_with_name": all_gt_ion_with_name, "PCCs": pearson_eval, "SA": short_angle,
                             "peptide": all_aa, "charge": all_charge, "length_peptide": all_len})
    ion_pred.to_json(os.path.join(output_dir, f"{instance_name}-IonIntensity_Pred_label.json"))

else:
    out_put = {}
    ion_pred = pd.DataFrame(
        {"pred_with_name": all_pred_ions_with_name,
         "peptide": all_aa, "charge": all_charge, "length_peptide": all_len})
    # ion_pred.to_hdf(os.path.join(output_dir, f"{instance_name}-IonIntensity_Pred_label.h5"), key='df', mode='w')
    for pred_ion, aa, charge in zip(all_pred_ions_with_name, all_aa, all_charge):
        out_put['%s.%d' % (aa, charge)] = pred_ion
    with open(os.path.join(output_dir, f"{instance_name}-IonIntensity_Pred_label.json"), 'w') as outfile:
        json.dump(out_put, outfile)
    # ion_pred.to_json(os.path.join(output_dir, f"{instance_name}-IonIntensity_Pred_label.json"))

if not No_Intensity:
    pearson_eval_median = np.median(pearson_eval)
    sa_eval_median = np.median(short_angle)
    logger.info(
        f'pearson_eval_median:{str(pearson_eval_median.item())}' + " " + f'sa_eval_median:{str(sa_eval_median.item())}')
    pearson_eval = np.array(pearson_eval)
    short_angle = np.array(short_angle)

    font = {'weight': 'normal',
            'size': 12, }
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7), dpi=300)

    hist_stat = ax1.hist(pearson_eval, bins=100)
    # axs.text(28, 0, f'mean: {sum(seq_len) / len(seq_len):.2f}', fontdict=font)
    ax1.set_ylabel("Frequency")
    ax1.set_xlabel("PCC")

    for i, j in zip(np.quantile(pearson_eval, [0.25, 0.50, 0.75]), [0.25, 0.50, 0.75]):
        ax1.axvline(x=i, color='r')

    quantiles_pcc = np.quantile(pearson_eval, [0.25, 0.5, 0.75])
    Statistics = [np.sum(pearson_eval > i) / len(pearson_eval) for i in [0.7, 0.8, 0.9]]
    ax1.text(0, 0.7,  f'>PCC Percentage\n>0.70 {Statistics[0]:.2%}\n>0.80 {Statistics[1]:.2%} \n>0.90 {Statistics[2]:.2%}\n'
                      f'PCC Quantile\n25% : {quantiles_pcc[0]:.3}\n50% : {quantiles_pcc[1]:.3}\n'
                      f'75% : {quantiles_pcc[2]:.3}\nN={len(pearson_eval)}\nAll Non-Ac Peptide={y_matrix_all.shape[0]}', fontdict=font, transform=ax1.transAxes)

    ax1.grid()
    fig.set_facecolor((1, 1, 1))
    ax1.set_title(f"PCC distribution")

    hist_stat = ax2.hist(short_angle, bins=100)
    ax2.set_ylabel("Frequency")
    ax2.set_xlabel("SA")

    for i, j in zip(np.quantile(short_angle, [0.25, 0.5, 0.75]), [0.25, 0.5, 0.75]):
        ax2.axvline(x=i, color='r')

    quantiles_sa = np.quantile(short_angle, [0.25, 0.5, 0.75])
    Statistics = [np.sum(short_angle > i) / len(short_angle) for i in [0.7, 0.8, 0.9]]
    ax2.text(0, 0.7,  f'>SA Percentage\n>0.70 {Statistics[0]:.2%}\n>0.80 {Statistics[1]:.2%} \n>0.90 {Statistics[2]:.2%}\n'
                      f'SA Quantile\n25% : {quantiles_sa[0]:.3}\n50% : {quantiles_sa[1]:.3}\n'
                      f'75% : {quantiles_sa[2]:.3}\nN={len(short_angle)}', fontdict=font, transform=ax2.transAxes)
    ax2.grid()
    ax2.set_title(f"SA distribution")
    fig.suptitle(f'{cfg.data_name} test result', fontsize=20)
    plt.savefig(os.path.join(output_dir, f"{instance_name}-Histogram.png"), dpi=300)
