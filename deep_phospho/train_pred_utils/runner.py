import copy
import datetime
import json
import os
import sys
import traceback
from os.path import join as join_path
from typing import Union

import torch

from deep_phospho import proteomics_utils as prot_utils
from deep_phospho.model_utils.logger import setup_logger
from deep_phospho.model_utils.param_config_load import load_config_from_json
from deep_phospho.train_pred_utils.ion_pred import pred_ion
from deep_phospho.train_pred_utils.ion_train import train_ion_model
from deep_phospho.train_pred_utils.rt_pred import pred_rt
from deep_phospho.train_pred_utils.rt_train import train_rt_model


class DeepPhosphoRunner(object):
    def __init__(self, args, start_time=None, msgs_for_arg_parsing=None, termin_flag=None):
        if start_time is None:
            self.start_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        else:
            self.start_time = start_time

        self.args: dict = args

        self.WorkDir: str = args['WorkDir']
        self.TaskName: str = args['TaskName']
        self.TrainData: Union[str, None] = args['Data-Train']
        self.TrainDataFormat: Union[str, None] = args['DataFormat-Train']
        self.PredData: list = args['Data-Pred']
        self.PredDataFormat: list = args['DataFormat-Pred']

        self.Train: bool = args['Task-Train']
        self.Predict: bool = args['Task-Predict']

        self.PretrainIonModel: str = args['PretrainModel-Ion']
        self.PretrainRTModel: dict[int: str] = args['PretrainModel-RT']

        self.SkipIonFinetune: bool = args['SkipFinetune-Ion']
        self.SkipRTFinetune: dict[int: bool] = args['SkipFinetune-RT']

        self.Device: str = args['Device']
        self.IonEpoch: int = args['Epoch-Ion']
        self.RTEpoch: int = args['Epoch-RT']
        self.IonBatchSize: int = args['BatchSize-Ion']
        self.RTBatchSize: int = args['BatchSize-RT']
        self.InitLR: float = args['InitLR']
        self.MaxPepLen: int = args['MaxPepLen']
        self.RTScale: (int, int) = args['RTScale']
        self.EnsembleRT: bool = args['EnsembleRT']
        self.rt_layers = [4, 5, 6, 7, 8] if self.EnsembleRT else [8]
        self.NoTime: bool = args['NoTime']
        self.Merge: bool = args['Merge']

        self.termin_flag = termin_flag

        self.config_dir, self.logger = self._init_workfolder(msgs_for_arg_parsing)
        self.train_data_path, self.pred_data_path = self._init_data_folder()

        if self.Train:
            self.trained_ion_model_folder = self.train_ion()
            self.trained_rt_model_folder = self.train_rt()

        if self.Predict:
            self.ion_pred_folders = self.pred_ion()
            self.rt_pred_folders = self.pred_rt()
            self.gen_lib()

    def _init_workfolder(self, msgs_for_arg_parsing):
        os.makedirs(self.WorkDir, exist_ok=True)
        logger = setup_logger('DeepPhosphoRunner', self.WorkDir, filename="RunnerLog.txt")
        if msgs_for_arg_parsing is not None:
            for m in msgs_for_arg_parsing:
                logger.info(m)

        with open(join_path(self.WorkDir, 'PassedArguments.txt'), 'w') as f:
            f.write(' '.join(sys.argv))
        with open(join_path(self.WorkDir, 'ParsedArguments.json'), 'w') as f:
            json.dump(self.args, f, indent=4)

        deep_phospho_main_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        if 'ConfigTemplate-Ion_model_train.json' in os.listdir(deep_phospho_main_dir):
            config_dir = deep_phospho_main_dir
        else:
            config_dir = join_path(deep_phospho_main_dir, 'Data', 'ConfigBackup')
        logger.info(f'Use config template stored in {config_dir}')
        return config_dir, logger

    def _init_data_folder(self):
        data_folder = join_path(self.WorkDir, 'Data')
        os.makedirs(data_folder, exist_ok=True)
        train_data_path = None
        if self.Train:
            if self.TrainData is not None and self.TrainData != '':
                self.logger.info(f'Transforming training data')
                train_data_path = prot_utils.dp_train_data.file_to_trainset(path=self.TrainData, output_folder=data_folder, file_type=self.TrainDataFormat)
                self.logger.info(f'Training data transformation done')
            else:
                self.logger.error('No training data is defined')
                raise FileNotFoundError

        pred_data_path = dict(IonPred=[], RTPred=[])
        if self.Predict:
            if self.TrainData is not None and self.TrainData != '':
                self.PredData.insert(0, self.TrainData)
                self.PredDataFormat.insert(0, self.TrainDataFormat)
            for idx, (pred_path, pred_type) in enumerate(zip(self.PredData, self.PredDataFormat), 1):
                print(idx, pred_path, pred_type)
                self.logger.info(f'Transforming prediction data {idx}: {pred_type} - {pred_path}')
                _ = prot_utils.dp_pred_data.file_to_pred_input(path=pred_path, output_folder=data_folder, file_type=pred_type)
                pred_data_path['IonPred'].append(_['IonPred'])
                pred_data_path['RTPred'].append(_['RTPred'])
                self.logger.info(f'Prediction data {idx} transformation done')

        return train_data_path, pred_data_path

    def _init_train_config_loading(self):
        if self.Train and (not self.SkipIonFinetune or not all(self.SkipRTFinetune.values())):
            self.logger.info('Init training')

        if self.SkipIonFinetune and ((self.SkipRTFinetune is True) or (self.EnsembleRT and isinstance(self.SkipRTFinetune, list) and len(self.SkipRTFinetune) == 5)):
            pass
        else:
            self.logger.info('Init training')

    def train_ion(self):
        trained_ion_model_folder = None
        if self.SkipIonFinetune:
            self.logger.info(f'Ion model training is skipped. Prediction will be performed with model {self.PretrainIonModel}')
        else:
            self.logger.info(f'Loading ion model config')
            ion_train_config = load_config_from_json(join_path(self.config_dir, 'ConfigTemplate-Ion_model_train.json'))
            ion_train_config['WorkFolder'] = self.WorkDir
            ion_train_config['Intensity_DATA_CFG']['TrainPATH'] = self.train_data_path['IonTrain']
            ion_train_config['Intensity_DATA_CFG']['TestPATH'] = self.train_data_path['IonVal']
            ion_train_config['Intensity_DATA_CFG']['HoldoutPATH'] = ''
            ion_train_config['Intensity_DATA_CFG']['DATA_PROCESS_CFG']['MAX_SEQ_LEN'] = self.MaxPepLen
            ion_train_config['TRAINING_HYPER_PARAM']['BATCH_SIZE'] = self.IonBatchSize
            ion_train_config['TRAINING_HYPER_PARAM']['TRAINING_HYPER_PARAM'] = self.InitLR
            ion_train_config['TRAINING_HYPER_PARAM']['EPOCH'] = self.IonEpoch
            ion_train_config['TRAINING_HYPER_PARAM']['GPU_INDEX'] = self.Device

            ion_train_config['PretrainParam'] = self.PretrainIonModel

            self.logger.info('-' * 20)
            self.logger.info('Start training ion intensity model')
            if self.NoTime:
                ion_train_config['InstanceName'] = f'{self.TaskName}-IonModel'
            else:
                ion_train_config['InstanceName'] = f'{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}-{self.TaskName}-IonModel'
            trained_ion_model_folder = join_path(ion_train_config['WorkFolder'], ion_train_config['InstanceName'])
            os.makedirs(trained_ion_model_folder, exist_ok=True)

            cfg_path = join_path(trained_ion_model_folder, f'Config-IonTrain-{self.TaskName}.json')
            with open(cfg_path, 'w') as f:
                json.dump(ion_train_config, f, indent=4)
            self.logger.info(f'Start ion model instance {ion_train_config["InstanceName"]}')
            try:
                stat = train_ion_model(ion_train_config, termin_flag=self.termin_flag)
                if stat is not None:
                    if stat == -1:
                        trained_ion_model_folder = None
                        print('runner stop')
            except Exception as e:
                error_msg = f'ERROR: Error when running ion model training instance {ion_train_config["InstanceName"]}'
                tb = traceback.format_exc()
                self.logger.error(error_msg + '\n' + tb)
                raise RuntimeError(error_msg)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        return trained_ion_model_folder

    def train_rt(self):
        total_rt_model_num = len(self.rt_layers)
        trained_rt_model_folder = {}
        for idx, layer in enumerate(self.rt_layers, 1):
            if self.SkipRTFinetune[layer]:
                self.logger.info(f'Training of RT model with layer {layer} is skipped. Prediction will be performed with model {self.PretrainRTModel[layer]}')
                continue

            self.logger.info(f'{idx}/{total_rt_model_num}: Init RT model training for layer {layer}')
            self.logger.info(f'Loading RT model config')
            rt_train_config = load_config_from_json(join_path(self.config_dir, 'ConfigTemplate-RT_model_train.json'))
            rt_train_config['WorkFolder'] = self.WorkDir
            rt_train_config['RT_DATA_CFG']['TrainPATH'] = self.train_data_path['RTTrain']
            rt_train_config['RT_DATA_CFG']['TestPATH'] = self.train_data_path['RTVal']
            rt_train_config['RT_DATA_CFG']['HoldoutPATH'] = ''
            rt_train_config['RT_DATA_CFG']['DATA_PROCESS_CFG']['MAX_SEQ_LEN'] = self.MaxPepLen
            rt_train_config['RT_DATA_CFG']['DATA_PROCESS_CFG']['MIN_RT'] = self.RTScale[0]
            rt_train_config['RT_DATA_CFG']['DATA_PROCESS_CFG']['MAX_RT'] = self.RTScale[1]
            rt_train_config['TRAINING_HYPER_PARAM']['BATCH_SIZE'] = self.RTBatchSize
            rt_train_config['TRAINING_HYPER_PARAM']['TRAINING_HYPER_PARAM'] = self.InitLR
            rt_train_config['TRAINING_HYPER_PARAM']['EPOCH'] = self.RTEpoch
            rt_train_config['TRAINING_HYPER_PARAM']['GPU_INDEX'] = self.Device

            rt_train_config['PretrainParam'] = self.PretrainRTModel[layer]
            rt_train_config['UsedModelCFG']['num_encd_layer'] = layer

            self.logger.info('-' * 20)
            self.logger.info(f'{idx}/{total_rt_model_num}: Start training RT model {layer}')
            if self.NoTime:
                rt_train_config['InstanceName'] = f'{self.TaskName}-RTModel-{layer}'
            else:
                rt_train_config['InstanceName'] = f'{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}-{self.TaskName}-RTModel-{layer}'
            trained_rt_model_folder[layer] = join_path(rt_train_config['WorkFolder'], rt_train_config['InstanceName'])
            os.makedirs(trained_rt_model_folder[layer], exist_ok=True)
            cfg_path = join_path(trained_rt_model_folder[layer], f'Config-RTTrain-{self.TaskName}-{layer}.json')
            with open(cfg_path, 'w') as f:
                json.dump(rt_train_config, f, indent=4)
            self.logger.info(f'Start rt model instance {rt_train_config["InstanceName"]}')
            try:
                stat = train_rt_model(configs=rt_train_config, termin_flag=self.termin_flag)
                if stat is not None:
                    if stat == -1:
                        print('runner stop')
                        break
            except Exception as e:
                error_msg = f'ERROR: Error when running rt model training instance {rt_train_config["InstanceName"]}'
                tb = traceback.format_exc()
                self.logger.error(error_msg + '\n' + tb)
                raise RuntimeError(error_msg)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        return trained_rt_model_folder

    def pred_ion(self):
        self.logger.info('Init ion intensity prediction')
        self.logger.info(f'Loading ion model config')
        ion_pred_config = load_config_from_json(join_path(self.config_dir, 'ConfigTemplate-Ion_model_pred.json'))
        ion_pred_config['WorkFolder'] = self.WorkDir
        if self.Train:
            if self.SkipIonFinetune:
                ion_pred_config['PretrainParam'] = self.PretrainIonModel
            else:
                ion_pred_config['PretrainParam'] = join_path(self.trained_ion_model_folder, 'ckpts', 'best_model.pth')
        else:
            ion_pred_config['PretrainParam'] = self.PretrainIonModel
        ion_pred_config['Intensity_DATA_CFG']['InputWithLabel'] = False
        ion_pred_config['Intensity_DATA_CFG']['DATA_PROCESS_CFG']['MAX_SEQ_LEN'] = self.MaxPepLen
        ion_pred_config['TRAINING_HYPER_PARAM']['BATCH_SIZE'] = self.IonBatchSize * 4
        ion_pred_config['TRAINING_HYPER_PARAM']['GPU_INDEX'] = self.Device
        ion_pred_folders = {}
        for idx, path in enumerate(self.pred_data_path['IonPred'], 1):
            self.logger.info(f'Init ion pred for {path}')
            self.logger.info(f'Use model parameters: {ion_pred_config["PretrainParam"]}')
            data_name = os.path.basename(path).replace('-Ion_PredInput.txt', '')
            cfg_cp = copy.deepcopy(ion_pred_config)
            cfg_cp['Intensity_DATA_CFG']['PredInputPATH'] = path
            if self.NoTime:
                cfg_cp['InstanceName'] = f'{self.TaskName}-IonPred-{data_name}'
            else:
                cfg_cp['InstanceName'] = f'{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}-{self.TaskName}-IonPred-{data_name}'
            ion_pred_folders[data_name] = (join_path(cfg_cp['WorkFolder'], cfg_cp['InstanceName']))
            os.makedirs(ion_pred_folders[data_name], exist_ok=True)
            cfg_path = join_path(ion_pred_folders[data_name], f'Config-IonPred-{self.TaskName}-{data_name}.json')
            with open(cfg_path, 'w') as f:
                json.dump(cfg_cp, f, indent=4)
            self.logger.info(f'Start ion model instance {cfg_cp["InstanceName"]}')
            try:
                stat = pred_ion(configs=cfg_cp, termin_flag=self.termin_flag)
                if stat is not None:
                    if stat == -1:
                        ion_pred_folders = None
                        print('runner stop')
                        break
            except:
                error_msg = f'ERROR: Error when running ion model prediction instance {cfg_cp["InstanceName"]}'
                tb = traceback.format_exc()
                self.logger.error(error_msg + '\n' + tb)
                raise RuntimeError(error_msg)
        return ion_pred_folders

    def pred_rt(self):
        self.logger.info(f'Loading RT model config')
        rt_pred_config = load_config_from_json(join_path(self.config_dir, 'ConfigTemplate-RT_model_pred.json'))
        rt_pred_config['WorkFolder'] = self.WorkDir
        rt_pred_config['RT_DATA_CFG']['InputWithLabel'] = False
        rt_pred_config['RT_DATA_CFG']['DATA_PROCESS_CFG']['MAX_SEQ_LEN'] = self.MaxPepLen
        rt_pred_config['RT_DATA_CFG']['DATA_PROCESS_CFG']['MIN_RT'] = self.RTScale[0]
        rt_pred_config['RT_DATA_CFG']['DATA_PROCESS_CFG']['MAX_RT'] = self.RTScale[1]
        rt_pred_config['TRAINING_HYPER_PARAM']['BATCH_SIZE'] = self.RTBatchSize * 4
        rt_pred_config['TRAINING_HYPER_PARAM']['GPU_INDEX'] = self.Device

        param_for_pred = dict()
        if self.Train:
            for layer in self.rt_layers:
                if self.SkipRTFinetune[layer]:
                    param_for_pred[layer] = self.PretrainRTModel[layer]
                else:
                    param_for_pred[layer] = join_path(self.trained_rt_model_folder[layer], 'ckpts', 'best_model.pth')
        else:
            param_for_pred = self.PretrainRTModel.copy()
        rt_pred_config['ParamsForPred'] = param_for_pred

        rt_pred_folders = {}
        for idx, path in enumerate(self.pred_data_path['RTPred'], 1):
            self.logger.info(f'Init RT pred for {path}')
            self.logger.info(f'Use model parameters: {rt_pred_config["ParamsForPred"]}')
            data_name = os.path.basename(path).replace('-RT_PredInput.txt', '')
            cfg_cp = copy.deepcopy(rt_pred_config)
            cfg_cp['RT_DATA_CFG']['PredInputPATH'] = path
            if self.NoTime:
                cfg_cp['InstanceName'] = f'{self.TaskName}-RTPred-{data_name}'
            else:
                cfg_cp['InstanceName'] = f'{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}-{self.TaskName}-RTPred-{data_name}'
            rt_pred_folders[data_name] = (join_path(cfg_cp['WorkFolder'], cfg_cp['InstanceName']))
            os.makedirs(rt_pred_folders[data_name], exist_ok=True)
            cfg_path = join_path(rt_pred_folders[data_name], f'Config-RTPred-{self.TaskName}-{data_name}.json')
            with open(cfg_path, 'w') as f:
                json.dump(cfg_cp, f, indent=4)
            self.logger.info(f'Start RT model instance {cfg_cp["InstanceName"]}')
            try:
                stat = pred_rt(configs=cfg_cp, termin_flag=self.termin_flag)
                if stat is not None:
                    if stat == -1:
                        rt_pred_folders = None
                        print('runner stop')
                        break
            except:
                error_msg = f'ERROR: Error when running rt model prediction instance {cfg_cp["InstanceName"]}'
                tb = traceback.format_exc()
                self.logger.error(error_msg + '\n' + tb)
                raise RuntimeError(error_msg)
        return rt_pred_folders

    def gen_lib(self):
        self.logger.info(f'Init library generation')
        lib_paths = {}
        for idx, (data_name, ion_result_folder) in enumerate(self.ion_pred_folders.items(), 1):
            self.logger.info(f'Generate library {data_name}')
            lib_path = prot_utils.gen_dp_lib.generate_spec_lib(
                data_name=data_name,
                output_folder=self.WorkDir,
                pred_ion_path=join_path(ion_result_folder, f'{os.path.basename(ion_result_folder)}-PredOutput.json'),
                pred_rt_path=join_path(self.rt_pred_folders[data_name], 'Prediction.txt'),
                logger=self.logger
            )
            lib_paths[data_name] = lib_path

        self.logger.info(f'Init library merging')
        main_key = list(lib_paths.keys())[0]
        path = prot_utils.gen_dp_lib.merge_lib(
            main_lib_path=lib_paths[main_key],
            add_libs_path={n: p for n, p in lib_paths.items() if n != main_key},
            output_folder=self.WorkDir,
            task_name=self.TaskName,
            logger=self.logger
        )

        self.logger.info(f'All finished. Check directory {self.WorkDir} for results.')
