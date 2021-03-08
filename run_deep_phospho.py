import os
from os.path import join as join_path
import sys
import copy
import datetime
import argparse
import json

from deep_phospho.model_utils.logger import setup_logger
from deep_phospho.model_utils.param_config_load import load_config_from_json
from deep_phospho import proteomics_utils as prot_utils
from deep_phospho.proteomics_utils import rapid_kit as rk


HelpMSG = '''
This script integrate all functions of DeepPhospho in one.
To run this, at least two files are needed:
    1. a file contains both spectra and (i)RT, and the following files are valid:
        I. spectral library from Spectronaut (either directDIA or DDA is fine)
        II. msms.txt file generated by MaxQuant
        [for more information about training data, please see -- and --]
    2. a file contains under-predicted peptide precursors, and the following files are valid:
        I. spectral library from Spectronaut (either directDIA or DDA)
        II. search result from Spectronaut
        III. msms.txt or evidence.txt file generated by MaxQuant
        IV. two-column tab-separated file with column name "sequence" and "charge"
        [for more information about prediction input data, please see -- and --]
We also provided some suggestions about the preperation of these two files, please visit our GitHub repository for more information.

The detailed help message about each argument is listed below
If you have any question or any suggestion, please contact us

At last, thank you for using DeepPhospho
[DeepPhospho repository] https://github.com/weizhenFrank/DeepPhospho

'''


def init_arg_parser():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter, description=HelpMSG)

    # work folder
    parser.add_argument('-w', '--work_dir', metavar='directory', type=str, default=None,
                        help='The task will start in this directory')
    # task name
    parser.add_argument('-t', '--task_name', metavar='str', type=str, default=None,
                        help='Task name will be used to name some folders')

    # train file
    parser.add_argument('-tf', '--train_file', metavar='path', type=str, required=True,
                        help='''Train file can be either a spectral library from Spectronaut or msms.txt from MaxQuant''')
    # train file type
    parser.add_argument('-tt', '--train_file_type', metavar='str', required=True,
                        help='''To use Spectronaut library, set this to "SNLib"
Use MaxQuant msms.txt file, set this to "MQ1.5" for MaxQuant version <= 1.5, and "MQ1.6" for version >= 1.6''')
    # pred file
    parser.add_argument('-pf', '--pred_file', metavar='path', type=str, required=True, nargs='*', action='append',
                        help='''File contains peptide precursors under-prediction may has multi-source:
I. spectral library from Spectronaut
II. search result from Spectronaut
III. msms.txt or evidence.txt from MaxQuant
IV. any tab-separated file with two columns "sequence" and "charge"''')
    # pred file type
    parser.add_argument('-pt', '--pred_file_type', metavar='str', type=str, required=True, nargs='*', action='append',
                        help='''The prediction file source or peptide format
I. "SNLib" for Spectronaut library
II. "SNResult" for Spectronaut result
III. "MQ1.5" or "MQ1.6" for msms.txt/evidence.txt from MaxQuant version <= 1.5 or >= 1.6
IV. for peptide list file, the modified peptides in the following format are valid
    a. "PepSN13" is Spectronaut 13+ peptide format like _[Acetyl (Protein N-term)]M[Oxidation (M)]LSLS[Phospho (STY)]PLK_
    b. "PepMQ1.5" is MaxQuant 1.5- peptide format like _(ac)GS(ph)QDM(ox)GS(ph)PLRET(ph)RK_
    c. "PepMQ1.6" is MaxQuant 1.6+ peptide format like _(Acetyl (Protein N-term))TM(Oxidation (M))DKS(Phospho (STY))ELVQK_
    d. "PepComet" is Comet peptide format like n#DFM*SPKFS@LT@DVEY@PAWCQDDEVPITM*QEIR
    e. "PepDP" is DeepPhospho used peptide format like *1ED2MCLK''')

    # device
    parser.add_argument('-d', '--device', metavar='cpu|0|1|...', type=str, default='cpu',
                        help='Use which device. This can be [cpu] or any integer (0, 1, 2, ...) to use corresponded GPU')
    # rt ensemble
    parser.add_argument('-en', '--rt_ensemble', default=False, action='store_true',
                        help='Use ensemble to improve RT prediction or not')
    # merge library
    parser.add_argument('-m', '--merge', default=False, action='store_true',
                        help='''To merge all predicted data to one library or not (the individual ones will also be kept)''')
    return parser


def create_folder(d):
    try:
        os.makedirs(d)
    except FileExistsError:
        pass


def parse_args(parser, time):
    inputs = copy.deepcopy(parser.parse_args().__dict__)
    arg_msgs = []

    work_dir = inputs['work_dir']
    if work_dir is None:
        work_dir = join_path(os.path.dirname(os.path.abspath(__file__)), f'{time}-DeepPhospho-WorkFolder')
        arg_msgs.append(f'-w or -work_dir is not passed, use {work_dir} as work directory')
    else:
        arg_msgs.append(f'Set work directory to {work_dir}')
    create_folder(work_dir)

    task_name = inputs['task_name']
    if task_name is None:
        task_name = f'Task_{time}'
        arg_msgs.append(f'-t or --task_name is not passed, use {task_name} as task name')
    else:
        task_name = f'{task_name}_{time}'
        arg_msgs.append(f'Set task name to {task_name}')

    train_file = os.path.abspath(inputs['train_file'])
    train_file_type = inputs['train_file_type']
    if not os.path.exists(train_file):
        raise FileNotFoundError(f'Train file not found - {train_file}')
    if train_file_type.lower() not in ['snlib', 'mq1.5', 'mq1.6']:
        raise ValueError(f'Train file type should be one of ["SNLib", "MQ1.5", "MQ1.6"], now {train_file_type}')
    arg_msgs.append(f'Train file with {train_file_type} format: {train_file}')

    pred_files = rk.sum_list(inputs['pred_file'])
    for file_idx, file in enumerate(pred_files, 1):
        if not os.path.exists(file):
            raise FileNotFoundError(f'Prediction file {file_idx} not found - {file}')

    pred_files_type = rk.sum_list(inputs['pred_file_type'])
    pred_file_type_num = len(pred_files_type)
    pred_file_num = len(pred_files)
    if pred_file_type_num != 1 and pred_file_type_num != pred_file_num:
        raise ValueError(f'Get {pred_file_num} prediction files but {pred_file_type_num} file type\n')
    elif pred_file_num != 1 and pred_file_type_num == 1:
        pred_files_type = pred_files_type * pred_file_num
        msg = (f'Get {pred_file_num} prediction files and 1 file type. '
               f'{pred_files_type[0]} will be assigned to all files\n')
    else:
        msg = f'Get {pred_file_num} prediction files and {pred_file_type_num} file type\n'
    files_str = '\n'.join(f'\t{t}: {file}' for t, file in zip(pred_files_type, pred_files))
    arg_msgs.append(f'{msg}{files_str}')

    device = inputs['device']
    arg_msgs.append(f'Set device to {device}')

    rt_ensemble = inputs['rt_ensemble']
    if rt_ensemble:
        arg_msgs.append(f'Use ensemble RT model')

    merge = inputs['merge']
    if merge:
        arg_msgs.append(f'Merge all predicted spectral libraries to one after prediction done')

    return arg_msgs, {
        'WorkDir': work_dir,
        'TaskName': task_name,
        'TrainData': (train_file, train_file_type),
        'PredData': list(zip(pred_files, pred_files_type)),
        'Device': device,
        'EnsembleRT': rt_ensemble,
        'Merge': merge
    }


if __name__ == '__main__':
    start_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    arg_parser = init_arg_parser()
    msgs, args = parse_args(arg_parser, start_time)

    WorkDir = args['WorkDir']
    TaskName = args['TaskName']
    TrainData = args['TrainData']
    PredData = args['PredData']
    Device = args['Device']
    EnsembleRT = args['EnsembleRT']
    Merge = args['Merge']

    logger = setup_logger('DeepPhosphoRunner', WorkDir, filename="RunnerLog.txt")
    for m in msgs:
        logger.info(m)

    with open(join_path(WorkDir, 'PassedArguments.txt'), 'w') as f:
        f.write(' '.join(sys.argv))
    with open(join_path(WorkDir, 'ParsedArguments.json'), 'w') as f:
        json.dump(args, f, indent=4)

    this_dir = os.path.dirname(os.path.abspath(__file__))
    if 'ConfigTemplate-Ion_model_train.json' in os.listdir(this_dir):
        config_dir = this_dir
    else:
        config_dir = join_path(this_dir, 'Data', 'ConfigBackup')
    logger.info(f'Use config template stored in {config_dir}')

    data_folder = join_path(WorkDir, 'Data')
    create_folder(data_folder)
    logger.info(f'Transforming training data')
    train_data_path = prot_utils.dp_train_data.file_to_trainset(path=TrainData[0], output_folder=data_folder, file_type=TrainData[1])
    logger.info(f'Training data transformation done')

    pred_data_path = dict(IonPred=[], RTPred=[])
    PredData.insert(0, (TrainData[0], TrainData[1]))
    for idx, (pred_path, pred_type) in enumerate(PredData, 1):
        logger.info(f'Transforming prediction data {idx}: {pred_type} - {pred_path}')
        _ = prot_utils.dp_pred_data.file_to_pred_input(path=pred_path, output_folder=data_folder, file_type=pred_type)
        pred_data_path['IonPred'].append(_['IonPred'])
        pred_data_path['RTPred'].append(_['RTPred'])
        logger.info(f'Prediction data {idx} transformation done')

    logger.info('Init training')
    logger.info(f'Loading ion model config')
    ion_train_config = load_config_from_json(join_path(config_dir, 'ConfigTemplate-Ion_model_train.json'))
    ion_train_config['WorkFolder'] = WorkDir
    ion_train_config['Intensity_DATA_CFG']['TrainPATH'] = train_data_path['IonTrain']
    ion_train_config['Intensity_DATA_CFG']['TestPATH'] = train_data_path['IonVal']
    ion_train_config['Intensity_DATA_CFG']['HoldoutPATH'] = ''
    ion_train_config['TRAINING_HYPER_PARAM']['GPU_INDEX'] = Device
    ion_train_config['TRAINING_HYPER_PARAM']['EPOCH'] = 1

    logger.info(f'Loading RT model config')
    rt_train_config = load_config_from_json(join_path(config_dir, 'ConfigTemplate-RT_model_train.json'))
    rt_train_config['WorkFolder'] = WorkDir
    rt_train_config['RT_DATA_CFG']['TrainPATH'] = train_data_path['RTTrain']
    rt_train_config['RT_DATA_CFG']['TestPATH'] = train_data_path['RTVal']
    rt_train_config['RT_DATA_CFG']['HoldoutPATH'] = ''
    rt_train_config['TRAINING_HYPER_PARAM']['EPOCH'] = 1
    rt_train_config['TRAINING_HYPER_PARAM']['GPU_INDEX'] = Device

    logger.info('-' * 20)
    logger.info('Start training ion intensity model')
    ion_train_config['InstanceName'] = f'{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}-{TaskName}-IonModel'
    ion_model_folder = join_path(ion_train_config['WorkFolder'], ion_train_config['InstanceName'])
    create_folder(ion_model_folder)
    cfg_path = join_path(ion_model_folder, f'Config-IonTrain-{TaskName}.json')
    with open(cfg_path, 'w') as f:
        json.dump(ion_train_config, f, indent=4)
    logger.info(f'Start ion model instance {ion_train_config["InstanceName"]}')
    code = os.system(f'python train_ion.py {cfg_path}')
    if code != 0:
        error_msg = f'Error when running ion model training instance {ion_train_config["InstanceName"]}'
        logger.error(error_msg)
        raise RuntimeError(error_msg)

    logger.info('-' * 20)
    logger.info('Start training RT model')
    if EnsembleRT:
        layer_nums = [4, 5, 6, 7, 8]
    else:
        layer_nums = [8]
    rt_model_folders = {}
    for idx, layer_num in enumerate(layer_nums, 1):
        logger.info(f'Training RT model {idx}')
        cfg_cp = copy.deepcopy(rt_train_config)
        cfg_cp['PretrainParam'] = f"./PretrainParams/RTModel/{layer_num}.pth"
        cfg_cp['UsedModelCFG']['num_encd_layer'] = layer_num
        cfg_cp['InstanceName'] = f'{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}-{TaskName}-RTModel-{layer_num}'
        rt_model_folders[layer_num] = (join_path(cfg_cp['WorkFolder'], cfg_cp['InstanceName']))
        create_folder(rt_model_folders[layer_num])
        cfg_path = join_path(rt_model_folders[layer_num], f'Config-RTTrain-{TaskName}-{layer_num}.json')
        with open(cfg_path, 'w') as f:
            json.dump(cfg_cp, f, indent=4)
        logger.info(f'Start rt model instance {cfg_cp["InstanceName"]}')
        code = os.system(f'python train_rt.py {cfg_path}')
        if code != 0:
            error_msg = f'Error when running rt model training instance {cfg_cp["InstanceName"]}'
            logger.error(error_msg)
            raise RuntimeError(error_msg)

    logger.info('Init prediction')
    logger.info(f'Loading ion model config')
    ion_pred_config = load_config_from_json(join_path(config_dir, 'ConfigTemplate-Ion_model_pred.json'))
    ion_pred_config['WorkFolder'] = WorkDir
    ion_pred_config['PretrainParam'] = join_path(ion_model_folder, 'ckpts', 'best_model.pth')
    ion_pred_config['Intensity_DATA_CFG']['InputWithLabel'] = False
    ion_pred_config['TRAINING_HYPER_PARAM']['GPU_INDEX'] = Device
    ion_pred_folders = {}
    for idx, path in enumerate(pred_data_path['IonPred'], 1):
        data_name = os.path.basename(path).replace('-Ion_PredInput.txt', '')
        cfg_cp = copy.deepcopy(ion_pred_config)
        cfg_cp['Intensity_DATA_CFG']['PredInputPATH'] = path
        cfg_cp['InstanceName'] = f'{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}-{TaskName}-IonPred-{data_name}'
        ion_pred_folders[data_name] = (join_path(cfg_cp['WorkFolder'], cfg_cp['InstanceName']))
        create_folder(ion_pred_folders[data_name])
        cfg_path = join_path(ion_pred_folders[data_name], f'Config-IonPred-{TaskName}-{data_name}.json')
        with open(cfg_path, 'w') as f:
            json.dump(cfg_cp, f, indent=4)
        logger.info(f'Start ion model instance {cfg_cp["InstanceName"]}')
        code = os.system(f'python pred_ion.py {cfg_path}')
        if code != 0:
            error_msg = f'Error when running ion model prediction instance {cfg_cp["InstanceName"]}'
            logger.error(error_msg)
            raise RuntimeError(error_msg)

    logger.info(f'Loading RT model config')
    rt_pred_config = load_config_from_json(join_path(config_dir, 'ConfigTemplate-RT_model_pred.json'))
    rt_pred_config['WorkFolder'] = WorkDir
    rt_pred_config['ParamsForPred'] = {str(layer_num): join_path(f, 'ckpts', 'best_model.pth') for layer_num, f in rt_model_folders.items()}
    rt_pred_config['RT_DATA_CFG']['InputWithLabel'] = False
    rt_pred_config['TRAINING_HYPER_PARAM']['GPU_INDEX'] = Device
    rt_pred_folders = {}
    for idx, path in enumerate(pred_data_path['RTPred'], 1):
        data_name = os.path.basename(path).replace('-RT_PredInput.txt', '')
        cfg_cp = copy.deepcopy(rt_pred_config)
        cfg_cp['RT_DATA_CFG']['PredInputPATH'] = path
        cfg_cp['InstanceName'] = f'{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}-{TaskName}-RTPred-{data_name}'
        rt_pred_folders[data_name] = (join_path(cfg_cp['WorkFolder'], cfg_cp['InstanceName']))
        create_folder(rt_pred_folders[data_name])
        cfg_path = join_path(rt_pred_folders[data_name], f'Config-RTPred-{TaskName}-{data_name}.json')
        with open(cfg_path, 'w') as f:
            json.dump(cfg_cp, f, indent=4)
        logger.info(f'Start RT model instance {cfg_cp["InstanceName"]}')
        code = os.system(f'python pred_rt.py {cfg_path}')
        if code != 0:
            error_msg = f'Error when running rt model prediction instance {cfg_cp["InstanceName"]}'
            logger.error(error_msg)
            raise RuntimeError(error_msg)

    logger.info(f'Init library generation')
    lib_paths = {}
    for idx, (data_name, ion_result_folder) in enumerate(ion_pred_folders.items(), 1):
        logger.info(f'Generate library {data_name}')
        lib_path = prot_utils.gen_dp_lib.generate_spec_lib(
            data_name=data_name,
            output_folder=WorkDir,
            pred_ion_path=join_path(ion_result_folder, f'{os.path.basename(ion_result_folder)}-PredOutput.json'),
            pred_rt_path=join_path(rt_pred_folders[data_name], 'Prediction.txt'),
            logger=logger
        )
        lib_paths[data_name] = lib_path

    logger.info(f'Init library merging')
    main_key = list(lib_paths.keys())[0]
    path = prot_utils.gen_dp_lib.merge_lib(
        main_lib_path=lib_paths[main_key],
        add_libs_path={n: p for n, p in lib_paths.items() if n != main_key},
        output_folder=WorkDir,
        task_name=TaskName,
        logger=logger
    )

    logger.info(f'All finished. Check directory {WorkDir} for results.')
