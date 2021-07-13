import copy
import datetime
import json
import os
import sys
from os.path import join as join_path

import torch

from deep_phospho import proteomics_utils as prot_utils
from deep_phospho.model_utils.logger import setup_logger
from deep_phospho.model_utils.param_config_load import load_config_from_json
from deep_phospho.proteomics_utils import rapid_kit as rk
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

        self.args = args

        self.WorkDir = args['WorkDir']
        self.TaskName = args['TaskName']
        self.TrainData = args['TrainData']
        self.PredData = args['PredData']

        self.ExistedIonModel = args['ExistedIonModel']
        self.ExistedRTModel = args['ExistedRTModel']
        self.SkipIonFinetune = args['SkipIonFinetune']
        self.SkipRTFinetune = args['SkipRTFinetune']

        self.Device = args['Device']
        self.IonEpoch = args['IonEpoch']
        self.RTEpoch = args['RTEpoch']
        self.IonBatchSize = args['IonBatchSize']
        self.RTBatchSize = args['RTBatchSize']
        self.InitLR = args['InitLR']
        self.MaxPepLen = args['MaxPepLen']
        self.RTScale = args['RTScale']
        self.EnsembleRT = args['EnsembleRT']
        self.NoTime = args['NoTime']
        self.Merge = args['Merge']

        self.termin_flag = termin_flag

        self.logger = self._init_logger(msgs_for_arg_parsing)
        self.config_dir = self._init_workfolder()
        self.train_data_path, self.pred_data_path = self._init_data_folder()
        if args['train']:
            self.ion_train_config, self.rt_train_config = self._init_train_config_loading()

        print(args['TrainMode'])
        print(f"train {args['train']} prediction {args['pred']}")
        if args['TrainMode'] == 'rt':
            if args['train']:
                self.used_predefined_rt_models, self.rt_model_folders = self.train_rt()
                if self.rt_model_folders is not None and args['pred']:
                    self.rt_pred_folders = self.pred_rt()
            elif args['pred']:
                self.rt_pred_folders = self.ExistedRTModel
                if self.rt_pred_folders is not None:
                    self.rt_pred_folders = self.pred_rt()
                else:
                    print('please give vaild model path.')
            else:
                print('do nothing')

        elif args['TrainMode'] == 'ion':
            if args['train']:
                self.ion_model_folder = self.train_ion()
                if args['pred'] and self.ion_model_folder is not None:
                    self.ion_pred_folders = self.pred_ion()
            elif args['pred']:
                # todo give model folder
                self.ion_model_folder = self.ExistedIonModel

                if self.ion_model_folder is not None:
                    self.ion_pred_folders = self.pred_ion()
                else:
                    print('please give vaild model path.')

        else:
            assert False

    def _init_logger(self, msgs_for_arg_parsing):
        os.makedirs(self.WorkDir, exist_ok=True)
        self.logger = setup_logger('DeepPhosphoRunner', self.WorkDir, filename="RunnerLog.txt")
        if msgs_for_arg_parsing is not None:
            for m in msgs_for_arg_parsing:
                self.logger.info(m)
        return self.logger

    def _init_workfolder(self):
        with open(join_path(self.WorkDir, 'PassedArguments.txt'), 'w') as f:
            f.write(' '.join(sys.argv))
        with open(join_path(self.WorkDir, 'ParsedArguments.json'), 'w') as f:
            json.dump(self.args, f, indent=4)

        deep_phospho_main_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        if 'ConfigTemplate-Ion_model_train.json' in os.listdir(deep_phospho_main_dir):
            config_dir = deep_phospho_main_dir
        else:
            config_dir = join_path(deep_phospho_main_dir, 'Data', 'ConfigBackup')
        self.logger.info(f'Use config template stored in {config_dir}')
        return config_dir

    def _init_data_folder(self):
        data_folder = join_path(self.WorkDir, 'Data')
        os.makedirs(data_folder, exist_ok=True)
        train_data_path = None
        if self.args['train']:
            if self.TrainData[0] is not None:
                self.logger.info(f'Transforming training data')
                train_data_path = prot_utils.dp_train_data.file_to_trainset(path=self.TrainData[0], output_folder=data_folder, file_type=self.TrainData[1])
                self.logger.info(f'Training data transformation done')
            else:
                train_data_path = None

        pred_data_path = dict(IonPred=[], RTPred=[])
        if self.args['pred']:
            # if self.TrainData[0] is not None:
            #     self.PredData.insert(0, (self.TrainData[0], self.TrainData[1]))
            assert len(self.PredData) > 0
            for idx, (pred_path, pred_type) in enumerate(self.PredData, 1):
                self.logger.info(f'Transforming prediction data {idx}: {pred_type} - {pred_path}')
                _ = prot_utils.dp_pred_data.file_to_pred_input(path=pred_path, output_folder=data_folder, file_type=pred_type)
                pred_data_path['IonPred'].append(_['IonPred'])
                pred_data_path['RTPred'].append(_['RTPred'])
                self.logger.info(f'Prediction data {idx} transformation done')

        return train_data_path, pred_data_path

    def _init_train_config_loading(self):
        if self.SkipIonFinetune and ((self.SkipRTFinetune is True) or (self.EnsembleRT and isinstance(self.SkipRTFinetune, list) and len(self.SkipRTFinetune) == 5)):
            pass
        else:
            self.logger.info('Init training')

        ion_train_config = dict()
        if self.SkipIonFinetune:
            self.logger.info(f'Existed ion model is passed. Ion model fine-tuning is skipped. {self.ExistedIonModel}')
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

        rt_train_config = dict()
        if self.EnsembleRT and (self.SkipRTFinetune is True or (isinstance(self.SkipRTFinetune, list) and len(self.SkipRTFinetune) == 5)):
            self.logger.info(f'Existed RT model is passed. RT model fine-tuning is skipped. {self.ExistedRTModel}')
        elif not self.EnsembleRT and self.SkipRTFinetune is True:
            self.logger.info(f'Existed RT model is passed. RT model fine-tuning is skipped. {self.ExistedRTModel[8]}')
        else:
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

        # rt_train_config['TRAINING_HYPER_PARAM']['DEBUG'] = True
        # ion_train_config['TRAINING_HYPER_PARAM']['DEBUG'] = True

        return ion_train_config, rt_train_config

    def train_ion(self):
        if not self.SkipIonFinetune:
            self.logger.info('-' * 20)
            self.logger.info('Start training ion intensity model')
            if self.NoTime:
                self.ion_train_config['InstanceName'] = f'{self.TaskName}-IonModel'
            else:
                self.ion_train_config['InstanceName'] = f'{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}-{self.TaskName}-IonModel'
            ion_model_folder = join_path(self.ion_train_config['WorkFolder'], self.ion_train_config['InstanceName'])
            os.makedirs(ion_model_folder, exist_ok=True)

            self.ion_train_config['PretrainParam'] = self.args['Pretrain-Ion']

            cfg_path = join_path(ion_model_folder, f'Config-IonTrain-{self.TaskName}.json')
            with open(cfg_path, 'w') as f:
                json.dump(self.ion_train_config, f, indent=4)
            self.logger.info(f'Start ion model instance {self.ion_train_config["InstanceName"]}')
            try:
                stat = train_ion_model(self.ion_train_config, termin_flag=self.termin_flag)
                if stat is not None:
                    if stat == -1:
                        ion_model_folder = None
                        print('runner stop')
            except:
                error_msg = f'ERROR: Error when running ion model training instance {self.ion_train_config["InstanceName"]}'
                self.logger.error(error_msg)
                raise RuntimeError(error_msg)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        else:
            ion_model_folder = None
        return ion_model_folder

    def train_rt(self):
        rt_model_folders = {}
        if self.SkipRTFinetune is True:
            if self.EnsembleRT:
                used_predefined_rt_models = self.ExistedRTModel.copy()
            else:
                used_predefined_rt_models = {8: self.ExistedRTModel[8]}
        else:
            if isinstance(self.SkipRTFinetune, list):
                used_predefined_rt_models = {l: self.ExistedRTModel[l] for l in self.SkipRTFinetune}
                layer_nums = rk.subtract_list([4, 5, 6, 7, 8], list(used_predefined_rt_models.keys()))
            else:
                used_predefined_rt_models = {}
                if self.EnsembleRT:
                    layer_nums = [4, 5, 6, 7, 8]
                else:
                    layer_nums = [8]

            self.logger.info('-' * 20)
            self.logger.info('Start training RT model')
            for idx, layer_idx in enumerate(layer_nums, 1):
                self.logger.info(f'Training RT model {idx}/{len(layer_nums)}')
                cfg_cp = copy.deepcopy(self.rt_train_config)
                cfg_cp['PretrainParam'] = self.args[f'Pretrain-RT-{layer_idx}']
                cfg_cp['UsedModelCFG']['num_encd_layer'] = layer_idx
                if self.NoTime:
                    cfg_cp['InstanceName'] = f'{self.TaskName}-RTModel-{layer_idx}'
                else:
                    cfg_cp['InstanceName'] = f'{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}-{self.TaskName}-RTModel-{layer_idx}'
                rt_model_folders[layer_idx] = (join_path(cfg_cp['WorkFolder'], cfg_cp['InstanceName']))
                os.makedirs(rt_model_folders[layer_idx], exist_ok=True)
                cfg_path = join_path(rt_model_folders[layer_idx], f'Config-RTTrain-{self.TaskName}-{layer_idx}.json')
                with open(cfg_path, 'w') as f:
                    json.dump(cfg_cp, f, indent=4)
                self.logger.info(f'Start rt model instance {cfg_cp["InstanceName"]}')
                try:
                    stat = train_rt_model(configs=cfg_cp, termin_flag=self.termin_flag)
                    if stat is not None:
                        if stat == -1:
                            rt_model_folders = None
                            print('runner stop')
                            break
                except:
                    error_msg = f'ERROR: Error when running rt model training instance {cfg_cp["InstanceName"]}'
                    self.logger.error(error_msg)
                    raise RuntimeError(error_msg)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        return used_predefined_rt_models, rt_model_folders

    def pred_ion(self):
        self.logger.info('Init prediction')
        self.logger.info(f'Loading ion model config')
        ion_pred_config = load_config_from_json(join_path(self.config_dir, 'ConfigTemplate-Ion_model_pred.json'))
        ion_pred_config['WorkFolder'] = self.WorkDir
        if self.SkipIonFinetune:
            ion_pred_config['PretrainParam'] = self.ExistedIonModel
        else:
            ion_pred_config['PretrainParam'] = join_path(self.ion_model_folder, 'ckpts', 'best_model.pth')
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
                self.logger.error(error_msg)
                raise RuntimeError(error_msg)
        return ion_pred_folders

    def pred_rt(self):
        self.logger.info(f'Loading RT model config')
        rt_pred_config = load_config_from_json(join_path(self.config_dir, 'ConfigTemplate-RT_model_pred.json'))
        rt_pred_config['WorkFolder'] = self.WorkDir
        rt_pred_config['ParamsForPred'] = {str(layer_num): join_path(f, 'ckpts', 'best_model.pth') for layer_num, f in self.rt_model_folders.items()}
        rt_pred_config['ParamsForPred'].update(self.used_predefined_rt_models)
        rt_pred_config['RT_DATA_CFG']['InputWithLabel'] = False
        rt_pred_config['RT_DATA_CFG']['DATA_PROCESS_CFG']['MAX_SEQ_LEN'] = self.MaxPepLen
        rt_pred_config['RT_DATA_CFG']['DATA_PROCESS_CFG']['MIN_RT'] = self.RTScale[0]
        rt_pred_config['RT_DATA_CFG']['DATA_PROCESS_CFG']['MAX_RT'] = self.RTScale[1]
        rt_pred_config['TRAINING_HYPER_PARAM']['BATCH_SIZE'] = self.RTBatchSize * 4
        rt_pred_config['TRAINING_HYPER_PARAM']['GPU_INDEX'] = self.Device
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
                self.logger.error(error_msg)
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
