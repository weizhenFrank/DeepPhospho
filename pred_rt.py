import os
import sys
import datetime
import random
import copy
from functools import partial
from tqdm import tqdm
import termcolor

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

import torch
from torch.utils.data import DataLoader

from deep_phospho.model_dataset.preprocess_input_data import RTdata, Dictionary
from deep_phospho.model_dataset.dataset import IonDataset, collate_fn

from deep_phospho.models.EnsembelModel import LSTMTransformer, LSTMTransformerEnsemble

from deep_phospho.model_utils.param_config_load import load_param_from_file, load_average_model, load_config_as_module, load_config_from_json
from deep_phospho.model_utils.logger import setup_logger
from deep_phospho.model_utils.loss_func import RMSELoss
from deep_phospho.model_utils.utils_functions import Delta_t95, Pearson, copy_files

from deep_phospho import proteomics_utils


# ---------------- User defined space Start --------------------

"""
Config file can be defined as
    a json file here
    or fill in the config_ion_model.py in DeepPhospho main folder
"""
config_path = r''
SEED = 666

# ---------------- User defined space End --------------------

this_script_dir = os.path.dirname(os.path.abspath(__file__))

if config_path != '':
    config_msg = f'Use config file path defined in train script: {config_path}'
elif len(sys.argv) == 2:
    config_path = sys.argv[1]
    config_msg = f'Use config file path defined in command line: {config_path}'
if config_path:
    configs = load_config_from_json(config_path)
    config_dir = os.path.dirname(config_path)
else:
    try:
        import config_rt_model as config_module
        config_path = os.path.join(this_script_dir, 'config_rt_model.py')
        config_msg = f'Use config_rt_model.py in DeepPhospho main folder as config file: {config_path}'
    except ModuleNotFoundError:
        from deep_phospho.configs import rt_config as config_module
        config_path = os.path.join(this_script_dir, 'deep_phospho', 'configs', 'rt_config.py')
        config_msg = f'Use default config file rt_config.py in DeepPhospho config module as config file'
    finally:
        configs = load_config_as_module(config_module)
        config_dir = this_script_dir


torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.backends.cudnn.deterministic = True
torch.autograd.set_detect_anomaly(True)


# Get data path here for ease of use
pred_input_file = configs['RT_DATA_CFG']['PredInputPATH']
if not os.path.exists(pred_input_file):
    raise FileNotFoundError(f'Input file is not existed in {pred_input_file}')

# Define task name as the specific identifier
task_info = (
    f'{configs["RT_DATA_CFG"]["DataName"]}'
    f'-{configs["UsedModelCFG"]["model_name"]}'
    f'-{configs["ExpName"]}'
)

init_time = datetime.datetime.now().strftime("%Y%m%d_%H_%M_%S")
instance_name = f'{init_time}-{task_info}'

# Get work folder and define output dir
work_folder = configs['WorkFolder']
if work_folder.lower() == 'here' or work_folder == '':
    work_folder = config_dir
output_dir = os.path.join(work_folder, instance_name)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Setup logger and add task info and the config msg
logger = setup_logger("RT", output_dir)
logger.info(f'Work folder is set to {work_folder}')
logger.info(f'Task start time: {init_time}')
logger.info(f'Task information: {task_info}')
logger.info(config_msg)

# Choose device (Set GPU index or default one, or use CPU)
if torch.cuda.is_available():
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


if configs['TRAINING_HYPER_PARAM']['loss_func'] == "MSE":
    loss_func = torch.nn.MSELoss()
elif configs['TRAINING_HYPER_PARAM']['loss_func'] == "RMSE":
    loss_func = RMSELoss()
elif configs['TRAINING_HYPER_PARAM']['loss_func'] == "L1":
    loss_func = torch.nn.L1Loss()
else:
    raise RuntimeError("no valid loss_func given")

dictionary = Dictionary()
idx2aa = dictionary.idx2word

RTtest = RTdata(configs, pred_input_file, dictionary=dictionary)
test_dataset = IonDataset(RTtest, configs)

test_dataloader = DataLoader(dataset=test_dataset,
                             shuffle=False,
                             batch_size=64 * 8,
                             num_workers=0,
                             collate_fn=partial(collate_fn, configs=configs))

rt_data = test_dataloader.dataset.ion_data


def idxtoaa(arr):
    peptide = [idx2aa[int(aa_idx)] for aa_idx in arr]
    return ''.join(peptide).replace('#', '').replace('$', '')


if configs['UsedModelCFG']['model_name'] == "LSTMTransformer":
    cfg_to_load = copy.deepcopy(configs['UsedModelCFG'])
    model = LSTMTransformer(
        # ntoken=Iontrain.N_aa,
        RT_mode=True,
        ntoken=RTtest.N_aa,
        # for prosit, it has 0-21
        **cfg_to_load,
    )

else:
    if configs['UsedModelCFG']['model_name'] == "LSTMTransformerEnsemble":

        model_arch_path = configs['ParamsForPred']

        models = []
        for arch, path in model_arch_path.items():
            cfg_to_load = copy.deepcopy(configs['UsedModelCFG'])

            Model = LSTMTransformer(
                RT_mode=True,
                ntoken=RTtest.N_aa,
                num_encd_layer=arch,  # change to 4, 6, 8, for model ensemble (original 8)
                **cfg_to_load,
            )

            Model = load_param_from_file(Model, path, partially=True, logger_name='RT')
            models.append(Model)

        model = LSTMTransformerEnsemble(models, RT_mode=True)

    else:
        raise RuntimeError("No valid model name given.")

if configs['UsedModelCFG']['model_name'] != "LSTMTransformerEnsemble":
    load_model_path = configs['ParamsForPred']
    load_model_path_dir = os.path.dirname(load_model_path)
    if configs['TEST_HYPER_PARAM']['Use multiple iteration']:
        model = load_average_model(model, load_model_path_dir, 10)

    else:
        model = load_param_from_file(model,
                                     load_model_path,
                                     partially=False, logger_name='RT')

copy_files(os.path.join(this_script_dir, 'deep_phospho', 'models', 'ion_model.py'), output_dir)
copy_files(os.path.join(this_script_dir, 'deep_phospho', 'models', 'EnsembelModel.py'), output_dir)
copy_files(os.path.join(this_script_dir, 'deep_phospho', 'models', 'auxiliary_loss_transformer.py'), output_dir)
copy_files(os.path.join(this_script_dir, 'pred_rt.py'), output_dir)
copy_files(config_path, output_dir)


model = model.to(device)

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
            seq_x = seq_x.to(device)
            x_hydro = x_hydro.to(device)
            x_rc = x_rc.to(device)
            pred_y = model(x1=seq_x, x2=x_hydro, x3=x_rc).squeeze()
        else:
            seq_x = inputs
            seq_x = seq_x.to(device)
            pred_y = model(x1=seq_x).squeeze()
        y = y.to(device)

        pred_ys.append(pred_y.detach().cpu())
        label_y.append(y.detach().cpu())
        seq_xs.append(seq_x.detach().cpu())

    pred_ys = torch.cat(pred_ys).numpy()
    label_y = torch.cat(label_y).numpy()
    seq_xs = torch.cat(seq_xs).numpy()

peptides = list(map(idxtoaa, seq_xs))

# torch.onnx.export(model,  # model being run
#                   inputs.to(device),  # model input (or a tuple for multiple inputs)
#                   os.path.join(output_dir, "test_2layers.onnx"),
#                   # where to save the model (can be a file or file-like object)
#                   export_params=True,  # store the trained parameter weights inside the model file
#                   opset_version=10,  # the ONNX version to export the model to
#                   do_constant_folding=True,  # whether to execute constant folding for optimization
#                   input_names=['input'],  # the model's input names
#                   output_names=['output'],  # the model's output names
#                   dynamic_axes={'input': {0: 'batch_size'},  # variable lenght axes
#                                 'output': {0: 'batch_size'}})


def re_norm(arr):
    arr = arr * (rt_data.MAX_RT - rt_data.MIN_RT) + rt_data.MIN_RT
    return arr


label_y = re_norm(label_y)
pred_ys = re_norm(pred_ys)

# ipdb.set_trace()


if configs['UsedModelCFG']['model_name'] == "LSTMTransformerEnsemble":
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
Output.to_csv(os.path.join(output_dir, 'Prediction.txt'), index_label=False, index=True, sep='\t')
# Output.to_csv(os.path.join(output_dir,
#                            f'Pred_{os.path.basename(load_model_path).split(".")[0]}.csv'), index=True, index_label=False)

# TODO This var controls whether with label or not, be True or False in Config (has been deleted)
WithLabel = configs['RT_DATA_CFG']['InputWithLabel']
if WithLabel:
    fig, ax = plt.subplots(1, 1, figsize=(5, 5), dpi=300)
    # ipdb.set_trace()
    proteomics_utils.plots.rt_reg(Output['label'].tolist(), Output['pred'].tolist(), ax=ax,
                     title=f'RT-{configs["data_name"]}-{configs["UsedModelCFG"]["model_name"]}',
                     save=f'{output_dir}/Plot')
