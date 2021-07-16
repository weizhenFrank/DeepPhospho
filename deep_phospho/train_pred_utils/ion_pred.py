import copy
import datetime
import json
import os
import random
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from deep_phospho.model_dataset.dataset import IonDataset, collate_fn
from deep_phospho.model_dataset.preprocess_input_data import IonData, Dictionary
from deep_phospho.model_utils.ion_eval import SA, Pearson
from deep_phospho.model_utils.logger import setup_logger, save_config
from deep_phospho.model_utils.param_config_load import load_param_from_file
from deep_phospho.model_utils.utils_functions import show_params_status, give_name_ion
from deep_phospho.models.EnsembelModel import LSTMTransformer
from deep_phospho.models.ion_model import StackedLSTM  # Use the LSTMTransformer in EnsembleModel.py

SEED = 666

torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.backends.cudnn.deterministic = True
torch.autograd.set_detect_anomaly(True)


def pred_ion(configs=None, config_load_msgs=None, config_overwrite_msgs=None, termin_flag=None):
    # Get data path here for ease of use
    pred_input_file = configs['Intensity_DATA_CFG']['PredInputPATH']
    if not os.path.exists(pred_input_file):
        raise FileNotFoundError(f'Input file is not existed in {pred_input_file}')

    # Define task name as the specific identifier
    task_info = (
        f'ion_inten-{configs["Intensity_DATA_CFG"]["DataName"]}'
        f'-{configs["UsedModelCFG"]["model_name"]}'
        f'-{configs["ExpName"]}'
        f'-remove_ac_pep{configs["TRAINING_HYPER_PARAM"]["remove_ac_pep"]}'
    )
    init_time = datetime.datetime.now().strftime("%Y%m%d_%H_%M_%S")

    if configs['InstanceName'] != '':
        instance_name = configs['InstanceName']
        instance_name_msg = f'Use manurally defined instance name {instance_name}'
    else:
        instance_name = f'{init_time}-{task_info}'
        instance_name_msg = f'No instance name defined in config or passed from arguments. Use {instance_name}'

    # Get work folder and define output dir
    work_folder = configs['WorkFolder']
    if work_folder.lower() == 'here' or work_folder == '':
        work_folder = '.'
    output_dir = os.path.join(work_folder, instance_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Setup logger and add task info and the config msg
    logger = setup_logger("IonInten", output_dir)
    logger.info(f'Work folder is set to {work_folder}')
    logger.info(f'Task start time: {init_time}')
    logger.info(f'Task information: {task_info}')
    logger.info(f'Instance name: {instance_name_msg}')
    if config_overwrite_msgs is not None:
        logger.info(config_overwrite_msgs)
    if config_load_msgs is not None:
        logger.info(config_load_msgs)

    logger.info(save_config(configs, output_dir))

    # Choose device (Set GPU index or default one, or use CPU)
    if configs["TRAINING_HYPER_PARAM"]['GPU_INDEX'].lower() == 'cpu':
        device = torch.device('cpu')
        logger.info(f'CPU is defined as the device in config')
        use_cuda = False
    elif torch.cuda.is_available():
        if configs["TRAINING_HYPER_PARAM"]['GPU_INDEX']:
            device = torch.device(f'cuda:{configs["TRAINING_HYPER_PARAM"]["GPU_INDEX"]}')
            logger.info(f'Cuda available. Use config defined GPU {configs["TRAINING_HYPER_PARAM"]["GPU_INDEX"]}')
        else:
            device = torch.device('cuda:0')
            logger.info(f'Cuda available. No GPU defined in config. Use "cuda:0"')
        use_cuda = True
    else:
        device = torch.device('cpu')
        logger.info(f'Cuda not available. Use CPU')
        use_cuda = False

    # Init dictionary
    dictionary = Dictionary()
    if 'X' in dictionary.word2idx:
        dictionary.idx2word.pop(dictionary.word2idx.pop('X'))

    idx2aa = dictionary.idx2word

    Iontest = IonData(configs, pred_input_file, dictionary=dictionary)
    test_dataset = IonDataset(Iontest, configs)
    test_dataloader = DataLoader(dataset=test_dataset,
                                 shuffle=False,
                                 batch_size=configs['TRAINING_HYPER_PARAM']['BATCH_SIZE'],
                                 num_workers=2,
                                 collate_fn=partial(collate_fn, configs=configs))

    def idxtoaa(arr):
        peptide = [idx2aa[int(aa_idx)] for aa_idx in arr]
        return ''.join(peptide).replace('#', '').replace('$', '')

    if configs['UsedModelCFG']['model_name'] == "StackedLSTM":
        cfg_to_load = copy.deepcopy(configs['UsedModelCFG'])
        model = StackedLSTM(
            # ntoken=Ionholdout.N_aa,
            ntoken=len(dictionary),
            # for prosit, it has 0-21
            # row_num=Ionholdout.row_num,
            row_num=53,
            # for prosit, it has 30 max length
            use_prosit=configs.data_name == 'Prosit',
            **cfg_to_load,
        )
    elif configs['UsedModelCFG']['model_name'] == "LSTMTransformer":
        cfg_to_load = copy.deepcopy(configs['UsedModelCFG'])
        model = LSTMTransformer(
            # ntoken=Iontrain.N_aa,
            RT_mode=False,
            ntoken=len(dictionary),
            # for prosit, it has 0-21
            use_prosit=configs['Intensity_DATA_CFG']["DataName"] == 'Prosit',
            pdeep2mode=configs['TRAINING_HYPER_PARAM']['pdeep2mode'],
            two_stage=configs['TRAINING_HYPER_PARAM']['two_stage'],
            **cfg_to_load,
        )
    else:
        raise Exception("No model given!")

    model = load_param_from_file(model,
                                 configs['PretrainParam'],
                                 partially=False, logger_name='IonInten')

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

    model = model.to(device)
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
            if termin_flag is not None:
                if termin_flag.qsize() > 0:
                    # set fales stop loop
                    return -1
            # ipdb.set_trace()
            if len(inputs) == 3:
                seq_x = inputs[0]
                x_charge = inputs[1]
                x_gb = inputs[2]
                seq_x = seq_x.to(device)
                x_charge = x_charge.to(device)
                x_gb = x_gb.to(device)
                pred_y = model(x1=seq_x, x2=x_charge, x3=x_gb)
            else:
                seq_x = inputs[0]
                x_charge = inputs[1]
                seq_x = seq_x.to(device)
                x_charge = x_charge.to(device)
                # ipdb.set_trace()
                # try:
                #     pred_y = model(x1=seq_x, x2=x_charge)
                # except RuntimeError:
                #     ipdb.set_trace()
                pred_y = model(x1=seq_x, x2=x_charge)
            y = y.to(device)
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

    WithLabel = configs['Intensity_DATA_CFG']['InputWithLabel']

    for pred_inten_mat, pep_inten_mat, aas, length_aas, charge in zip(pred_matrix_all, y_matrix_all, pep, pep_len, charges):

        if WithLabel:
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
    if WithLabel:
        ion_pred = pd.DataFrame({"IntPep": all_aa, "PrecCharge": all_charge, "PepLen": all_len,
                                 "gt_ion_with_name": all_gt_ion_with_name, "PredInten": all_pred_ions_with_name,
                                 "PCCs": pearson_eval, "SA": short_angle})
        ion_pred.to_json(os.path.join(output_dir, f"{instance_name}-PredOutput.json"))

    else:
        out_put = {}
        ion_pred = pd.DataFrame(
            {"IntPep": all_aa, "PrecCharge": all_charge, "PepLen": all_len, "PredInten": all_pred_ions_with_name})
        # ion_pred.to_hdf(os.path.join(output_dir, f"{instance_name}-IonIntensity_Pred_label.h5"), key='df', mode='w')
        for pred_ion, aa, charge in zip(all_pred_ions_with_name, all_aa, all_charge):
            out_put['%s.%d' % (aa, charge)] = pred_ion
        with open(os.path.join(output_dir, f"{instance_name}-PredOutput.json"), 'w') as outfile:
            json.dump(out_put, outfile, indent=4)
        # ion_pred.to_json(os.path.join(output_dir, f"{instance_name}-IonIntensity_Pred_label.json"))

    if WithLabel:
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
        ax1.text(0, 0.7, f'>PCC Percentage\n>0.70 {Statistics[0]:.2%}\n>0.80 {Statistics[1]:.2%} \n>0.90 {Statistics[2]:.2%}\n'
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
        ax2.text(0, 0.7, f'>SA Percentage\n>0.70 {Statistics[0]:.2%}\n>0.80 {Statistics[1]:.2%} \n>0.90 {Statistics[2]:.2%}\n'
                         f'SA Quantile\n25% : {quantiles_sa[0]:.3}\n50% : {quantiles_sa[1]:.3}\n'
                         f'75% : {quantiles_sa[2]:.3}\nN={len(short_angle)}', fontdict=font, transform=ax2.transAxes)
        ax2.grid()
        ax2.set_title(f"SA distribution")
        fig.suptitle(f'{configs["Intensity_DATA_CFG"]["DataName"]} test result', fontsize=20)
        plt.savefig(os.path.join(output_dir, f"{instance_name}-Histogram.png"), dpi=300)
