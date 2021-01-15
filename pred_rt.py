import os
import argparse
import datetime
import numpy as np
import pandas as pd
import copy
import termcolor

from matplotlib import pyplot as plt
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from deep_phospho.configs import config_main as cfg
from deep_phospho.model_dataset.preprocess_input_data import RTdata, Dictionary

from deep_phospho.model_utils.param_config_load import load_param_from_file, load_average_model
from deep_phospho.model_utils.logger import setup_logger
from deep_phospho.model_utils.loss_func import RMSELoss

from deep_phospho.model_dataset.dataset import IonDataset, collate_fn
from deep_phospho.models.EnsembelModel import  LSTMTransformer, LSTMTransformerEnsemble


from deep_phospho.model_utils.utils_functions import Delta_t95, Pearson, copy_files
import random
import sys

sys.path.insert(0, "deep_phospho/bioplotkit")
sys.path.insert(0, "deep_phospho/mskit")

import bioplotkit as bpk
import ipdb

SEED = 666
torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.backends.cudnn.deterministic = True
torch.autograd.set_detect_anomaly(True)

logger = setup_logger("RT", None)


def get_parser():
    parser = argparse.ArgumentParser(description='RT prediction Analysis')
    parser.add_argument('--exp_name', type=str, default='', help="expriments name for output dir")
    parser.add_argument('--GPU', type=int, default=None, help="index of GPU")
    parser.add_argument('--pretrained_model', default=None, type=str, help="path of pretrained_model")
    parser.add_argument('--ad_hoc', default=None, help="ad_hoc operation")
    return parser.parse_args()


args = get_parser()

if args.GPU is not None:
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.GPU)
else:

    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.TRAINING_HYPER_PARAM['GPU_INDEX']

comment = f'RT-{cfg.data_name}-{cfg.MODEL_CFG["model_name"]}-{args.exp_name}'
now = datetime.datetime.now()
time_str = now.strftime("%Y_%m_%d_%H_%M_%S")
instance_name = f'{comment}-{time_str}'
output_dir = os.path.join('../result/RT/Analysis', instance_name)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

logger = setup_logger("RT", output_dir)

if cfg.TRAINING_HYPER_PARAM['loss_func'] == "MSE":
    loss_func = torch.nn.MSELoss()
elif cfg.TRAINING_HYPER_PARAM['loss_func'] == "RMSE":
    loss_func = RMSELoss()
elif cfg.TRAINING_HYPER_PARAM['loss_func'] == "L1":
    loss_func = torch.nn.L1Loss()
else:
    raise RuntimeError("no valid loss_func given")

dictionary = Dictionary(
    path="../data/20200724-Jeff-MQ_Author-MaxScore_Spec.json")

idx2aa = dictionary.idx2word

RTtest = RTdata(cfg.HOLDOUT_DATA_CFG, dictionary=dictionary)

test_dataset = IonDataset(RTtest)

test_dataloader = DataLoader(dataset=test_dataset,
                             shuffle=False,
                             batch_size=64 * 8,
                             num_workers=0,
                             collate_fn=collate_fn)

rt_data = test_dataloader.dataset.ion_data


def idxtoaa(arr):
    peptide = [idx2aa[int(aa_idx)] for aa_idx in arr]
    return ''.join(peptide).replace('#', '').replace('$', '')


if cfg.MODEL_CFG['model_name'] == "LSTMTransformer":
    cfg_to_load = copy.deepcopy(cfg.MODEL_CFG)
    model = LSTMTransformer(
        # ntoken=Iontrain.N_aa,
        RT_mode=cfg.Mode == "RT" or cfg.Mode == "Detect",
        ntoken=RTtest.N_aa,
        # for prosit, it has 0-21
        **cfg_to_load,
    )

else:
    if cfg.MODEL_CFG['model_name'] == "LSTMTransformerEnsemble":

        model_arch_path = {

            4: "/p300/projects/IonAndRT/result/RT/AcData/2020-10-24_11-28-37_PhosDIA_DIA18_finetune_RT-LSTMTransformer-4_RemoveSigmoidBitRTRange/ckpts/best_model.pth",
            5: "/p300/projects/IonAndRT/result/RT/AcData/2020-10-24_11-28-37_PhosDIA_DIA18_finetune_RT-LSTMTransformer-5_RemoveSigmoidBitRTRange/ckpts/best_model.pth",
            6: "/p300/projects/IonAndRT/result/RT/AcData/2020-10-24_11-28-37_PhosDIA_DIA18_finetune_RT-LSTMTransformer-6_RemoveSigmoidBitRTRange/ckpts/best_model.pth",
            7: "/p300/projects/IonAndRT/result/RT/AcData/2020-10-24_11-28-37_PhosDIA_DIA18_finetune_RT-LSTMTransformer-7_RemoveSigmoidBitRTRange/ckpts/best_model.pth",
            8: "/p300/projects/IonAndRT/result/RT/AcData/2020-10-24_11-28-37_PhosDIA_DIA18_finetune_RT-LSTMTransformer-8_RemoveSigmoidBitRTRange/ckpts/best_model.pth",
        }

        models = []
        for arch, path in model_arch_path.items():
            cfg_to_load = copy.deepcopy(cfg.MODEL_CFG)

            Model = LSTMTransformer(
                RT_mode=cfg.Mode == "RT" or cfg.Mode == "Detect",
                ntoken=RTtest.N_aa,
                num_encd_layer=arch,  # change to 4, 6, 8, for model ensemble (original 8)
                **cfg_to_load,
            )

            Model = load_param_from_file(Model, path, partially=True, logger_name='RT')
            models.append(Model)

        model = LSTMTransformerEnsemble(models, RT_mode=cfg.Mode == "RT" or cfg.Mode == "Detect")

    else:
        raise RuntimeError("No valid model name given.")

if not cfg.MODEL_CFG['model_name'] == "LSTMTransformerEnsemble":
    # load_model_path = "/p300/projects/IonAndRT/result/RT/AcData/2020-09-11_23-13-51_DDA_RT-LSTMTransformer-SameTrainValLength_Jeff_pretrain/ckpts/30600.pth"
    load_model_path = "/p300/projects/IonAndRT/result/RT/AcData/2020-10-11_22-20-05_R2P2_RT-LSTMTransformer-HumanJeffVeroE6_Correct_remove_outlier_2/ckpts/best_model.pth"
    load_model_path_dir = os.path.dirname(load_model_path)
    if cfg.TEST_HYPER_PARAM['Use multiple iteration']:
        model = load_average_model(model, load_model_path_dir, 10)

    else:
        model = load_param_from_file(model,
                                     load_model_path,
                                     partially=False, logger_name='RT')

copy_files("deep_phospho/models/ion_model.py", output_dir)
copy_files("deep_phospho/models/EnsembelModel.py", output_dir)
copy_files("pred_rt.py", output_dir)
copy_files("deep_phospho/configs", output_dir)

if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    model = torch.nn.DataParallel(model)
model = model.cuda()

model.eval()
pred_ys = []
label_y = []
seq_xs = []

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

        if isinstance(inputs, tuple):
            seq_x, x_hydro, x_rc = inputs
            seq_x = seq_x.cuda()
            x_hydro = x_hydro.cuda()
            x_rc = x_rc.cuda()
            pred_y = model(x1=seq_x, x2=x_hydro, x3=x_rc).squeeze()
        else:
            seq_x = inputs
            seq_x = seq_x.cuda()
            pred_y = model(x1=seq_x).squeeze()
        y = y.cuda()

        pred_ys.append(pred_y.detach().cpu())
        label_y.append(y.detach().cpu())
        seq_xs.append(seq_x.detach().cpu())

    pred_ys = torch.cat(pred_ys).numpy()
    label_y = torch.cat(label_y).numpy()
    seq_xs = torch.cat(seq_xs).numpy()

peptides = list(map(idxtoaa, seq_xs))

ipdb.set_trace()
torch.onnx.export(model,  # model being run
                  inputs.cuda(),  # model input (or a tuple for multiple inputs)
                  os.path.join(output_dir, "test_2layers.onnx"),
                  # where to save the model (can be a file or file-like object)
                  export_params=True,  # store the trained parameter weights inside the model file
                  opset_version=10,  # the ONNX version to export the model to
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names=['input'],  # the model's input names
                  output_names=['output'],  # the model's output names
                  dynamic_axes={'input': {0: 'batch_size'},  # variable lenght axes
                                'output': {0: 'batch_size'}})


def re_norm(arr):
    arr = arr * (rt_data.MAX_RT - rt_data.MIN_RT) + rt_data.MIN_RT
    return arr


label_y = re_norm(label_y)
pred_ys = re_norm(pred_ys)

# ipdb.set_trace()


if cfg.MODEL_CFG['model_name'] == "LSTMTransformerEnsemble":
    for idx in range(len(model_arch_path)):
        pearson_eval_unnormed = Pearson(label_y, pred_ys[:, idx])
        delta_t95_eval_unnormed = Delta_t95(label_y, pred_ys[:, idx])
        # ipdb.set_trace()
        logger.info(
            termcolor.colored(
                "{idx}-model:\npearson_eval_unnormed:{pearson_eval_unnormed:.5}\ndelta_t95_eval_unnormed:{delta_t95_eval_unnormed:.5}".format(
                    idx=idx,
                    pearson_eval_unnormed=pearson_eval_unnormed,
                    delta_t95_eval_unnormed=delta_t95_eval_unnormed,
                ), "green"))
        print()
    pred_ys = pred_ys.mean(axis=-1)

pearson_eval_unnormed = Pearson(label_y, pred_ys)
delta_t95_eval_unnormed = Delta_t95(label_y, pred_ys)

logger.info(
    termcolor.colored(
        "final result:\npearson_eval_unnormed:{pearson_eval_unnormed:.5}\ndelta_t95_eval_unnormed:{delta_t95_eval_unnormed:.5}".format(
            pearson_eval_unnormed=pearson_eval_unnormed,
            delta_t95_eval_unnormed=delta_t95_eval_unnormed,
        ), "green")
)

Output = pd.DataFrame({'sequence': peptides, 'pred': pred_ys, 'label': label_y})
Output.to_csv(os.path.join(output_dir, 'Prediction.csv'), index_label=False, index=True)
# Output.to_csv(os.path.join(output_dir,
#                            f'Pred_{os.path.basename(load_model_path).split(".")[0]}.csv'), index=True, index_label=False)

# TODO This var controls whether with label or not, be True or False in Config (has been deleted)
No_RT_gt = cfg.HOLDOUT_DATA_CFG['To_Predict']
if No_RT_gt:
    fig, ax = plt.subplots(1, 1, figsize=(7, 7), dpi=300)
    # ipdb.set_trace()
    bpk.plots.rt_reg(Output['label'].tolist(), Output['pred'].tolist(), ax=ax,
                     title=f'RT-{cfg.data_name}-{cfg.MODEL_CFG["model_name"]}',
                     save=f'{output_dir}/Plot')
