import logging
import os
import queue
import subprocess
import threading
import traceback

import wx

from deep_phospho import proteomics_utils as prot_utils
from deep_phospho.proteomics_utils import rapid_kit as rk
from deep_phospho.train_pred_utils.runner import DeepPhosphoRunner


def search_pretrain_params(ui_config):
    _this = os.path.abspath(__file__)
    dp_main_dir = os.path.dirname(os.path.dirname(os.path.dirname(_this)))
    pretrain_model_paths = rk.fill_path_dict(os.path.join(dp_main_dir, 'PretrainParams', '{}', '{}'), {
        'Pretrain-Ion': ['IonModel', 'best_model.pth'],
        'Pretrain-RT-4': ['RTModel', '4.pth'],
        'Pretrain-RT-5': ['RTModel', '5.pth'],
        'Pretrain-RT-6': ['RTModel', '6.pth'],
        'Pretrain-RT-7': ['RTModel', '7.pth'],
        'Pretrain-RT-8': ['RTModel', '8.pth'],
    })
    for k, p in pretrain_model_paths.items():
        if os.path.exists(p):
            ui_config[k] = p
    return ui_config


def search_pretrain_params_all(ui_config, search_folder='all'):
    ui_config = search_pretrain_params(ui_config)
    if any([ui_config[_] == '' for _ in ['Pretrain-Ion', 'Pretrain-RT-4', 'Pretrain-RT-5', 'Pretrain-RT-6', 'Pretrain-RT-7', 'Pretrain-RT-8']]):
        _this = os.path.abspath(__file__)
        dp_main_dir = os.path.dirname(os.path.dirname(os.path.dirname(_this)))
        if search_folder == 'all':
            for folder, dirs, non_dirs in os.walk(dp_main_dir):
                for non_dir in non_dirs:
                    if non_dir == 'best_model.pth':
                        ui_config['Pretrain-Ion'] = os.path.join(folder, non_dir)
                    else:
                        pass


def check_ui_config(config_from_ui: dict) -> (int, str, dict):
    """
    :return: tuple of (error code, message, config dict)
        error code: 1 for passed check and -1 with error
        message: string, and empty string if no message
        config dict: Checked config, with
    """
    if config_from_ui['Task-Train'] is False and config_from_ui['Task-Predict'] is False:
        return -1, 'At least one task of "Train" and "Predict" should be checked', config_from_ui

    # Pre-trained model
    _pretrain_ion = config_from_ui['Pretrain-Ion']
    if _pretrain_ion is None:
        config_from_ui['Pretrain-Ion'] = ''
    if _pretrain_ion != '' and not os.path.exists(_pretrain_ion):
        return -1, f'Pre-trained ion model parameter file {_pretrain_ion} is not existed', config_from_ui

    assert isinstance(config_from_ui['EnsembleRT'], bool)
    if config_from_ui['EnsembleRT']:
        rt_layers = [4, 5, 6, 7, 8]
    else:
        rt_layers = [8]
    for l in rt_layers:
        _pretrain_rt = config_from_ui[f'Pretrain-RT-{l}']
        if _pretrain_rt is None:
            config_from_ui[f'Pretrain-RT-{l}'] = ''
        if _pretrain_rt != '' and not os.path.exists(_pretrain_rt):
            return -1, f'Pre-trained RT model parameter file {_pretrain_rt} is not existed', config_from_ui

    # Device
    _device = config_from_ui['Device'].lower()
    if _device != 'cpu' and not _device.isdigit():
        return -1, f'Device should be "cpu" or the GPU index like "0", "1", "2", ... Now {_device}', config_from_ui
    config_from_ui['Device'] = _device

    # No need to check: RTScale-lower, RTScale-upper, MaxPepLen

    # Training data
    if config_from_ui['Task-Train']:
        _train_path = config_from_ui['TrainData']
        if _train_path is None or _train_path == '':
            return -1, f'Training data file {_train_path} is not defined and training task is on', config_from_ui
        elif not os.path.exists(_train_path):
            return -1, f'Training data file {_train_path} can not be found', config_from_ui

    # Init LR
    _init_lr = float(config_from_ui['InitLR'])
    if _init_lr >= 1:
        return -1, f'Please define a learning rate. Now {_init_lr}'

    # Prediction data
    if config_from_ui['Task-Predict']:
        checked_pred_data_paths = []
        checked_pred_data_formats = []
        for _pred_data_path, _pred_data_format in zip(config_from_ui['PredInput'], config_from_ui['PredInputFormat']):
            if _pred_data_path is None or _pred_data_path == '':
                pass
            elif not os.path.exists(_pred_data_path):
                return -1, f'Prediction input file {_pred_data_path} is defined but can not be found', config_from_ui
            else:
                checked_pred_data_paths.append(_pred_data_path)
                checked_pred_data_formats.append(_pred_data_format)
        if len(checked_pred_data_paths) == 0:
            return -1, f'Prediction task is set on, but no prediction input files can be found', config_from_ui
        config_from_ui['PredInput'] = checked_pred_data_paths
        config_from_ui['PredInputFormat'] = checked_pred_data_formats

    return 1, '', config_from_ui


def fillin_runner_cmd_from_ui_config(config_from_ui: dict) -> str:
    dp_main_folder = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    runner_cmd_parts = [
        'python',
        os.path.join(dp_main_folder, 'run_deep_phospho.py'),
        f'-w {config_from_ui["WorkFolder"]}',
        f'-t {config_from_ui["TaskName"]}',
    ]

    if config_from_ui['TrainData'] != '' and config_from_ui['TrainData'] is not None:
        runner_cmd_parts.append(f'-tf {config_from_ui["TrainData"]}')
        runner_cmd_parts.append(f'-tt {config_from_ui["TrainDataFormat"]}')

    for pred_file, pred_format in zip(config_from_ui['PredInput'], config_from_ui['PredInputFormat']):
        runner_cmd_parts.append(f'-pf {pred_file} -pt {pred_format}')

    runner_cmd_parts.append(f'-d {config_from_ui["Device"]}')
    runner_cmd_parts.append(f'-ie {config_from_ui["Epoch-Ion"]}')
    runner_cmd_parts.append(f'-re {config_from_ui["Epoch-RT"]}')

    runner_cmd_parts.append(f'-ibs {config_from_ui["BatchSize-Ion"]}')
    runner_cmd_parts.append(f'-rbs {config_from_ui["BatchSize-RT"]}')
    runner_cmd_parts.append(f'-lr {config_from_ui["InitLR"]}')
    runner_cmd_parts.append(f'-ml {config_from_ui["MaxPepLen"]}')
    runner_cmd_parts.append(f'-rs *{config_from_ui["RTScale-lower"]},{config_from_ui["RTScale-upper"]}')

    if config_from_ui['EnsembleRT']:
        runner_cmd_parts.append(f'-en')

    if config_from_ui['Task-Train']:
        runner_cmd_parts.append(f'-train 1')
    else:
        runner_cmd_parts.append(f'-train 0')

    if config_from_ui['Task-Predict']:
        runner_cmd_parts.append(f'-pred 1')
    else:
        runner_cmd_parts.append(f'-pred 0')

    runner_cmd_parts.append(f'-pretrain_ion {config_from_ui["Pretrain-Ion"]}')
    for l in [4, 5, 6, 7, 8]:
        runner_cmd_parts.append(f'-pretrain_rt_{l} {config_from_ui[f"Pretrain-RT-{l}"]}')

    runner_cmd_parts.append(f'-m')

    return ' '.join(runner_cmd_parts)


class CMDRunnerThread(threading.Thread):
    def __init__(self, window, *args, **kwargs):
        super(CMDRunnerThread, self).__init__(*args, **kwargs)
        self.window = window

        self.runner_process = None
        self.cmd = None
        self.terminated = False

    def set_cmd(self, cmd):
        self.cmd = cmd

    def run(self):
        self.terminated = False
        try:
            self.runner_process = subprocess.Popen(self.cmd)
            self.runner_process.wait()
            if self.terminated:
                pass
            elif self.runner_process.poll() is None:
                raise ValueError('Still running')
            elif self.runner_process.poll() == 0:
                wx.CallAfter(self.window.running_done)
            else:
                wx.CallAfter(self.window.running_error)
        except Exception:
            tb = traceback.format_exc()
            print(tb)
            wx.CallAfter(self.window.running_error)
        finally:
            self.terminated = False

    def terminate(self):
        self.runner_process.terminate()
        self.terminated = True
        wx.CallAfter(self.window.running_cancel)


class BuildLibThread(threading.Thread):
    def __init__(self, window, ion_input, rt_input, output_lib, *args, **kwargs):
        super(BuildLibThread, self).__init__(*args, **kwargs)
        self.window = window
        # TODO check input files and raise error to UI if has
        self.ion_input = ion_input
        self.rt_input = rt_input
        self.output_lib = output_lib
        self._running = True

    def run(self):
        try:
            prot_utils.gen_dp_lib.generate_spec_lib(
                data_name=None,
                output_folder=None,
                pred_ion_path=self.ion_input,
                pred_rt_path=self.rt_input,
                save_path=self.output_lib,
                logger=logging.getLogger(name='Build library')
            )
            wx.CallAfter(self.window.build_done)
        except:
            err_msg = f''
            wx.CallAfter(self.window.build_error, err_msg)


class MergeLibThread(threading.Thread):
    def __init__(self, window, input_lib_paths, output_lib, *args, **kwargs):
        super(MergeLibThread, self).__init__(*args, **kwargs)
        self.window = window
        self.input_lib_paths = input_lib_paths  # TODO check input lib files and raise error to UI if has
        self.output_lib = output_lib
        self._running = True

    def run(self):
        try:
            prot_utils.gen_dp_lib.merge_lib(
                main_lib_path=self.input_lib_paths[0],
                add_libs_path={f'Addtional library {i}': p for i, p in enumerate(self.input_lib_paths[1:], 1)},
                output_folder=None,
                task_name=None,
                save_path=self.output_lib,
                logger=logging.getLogger(name='Merge library')
            )
            wx.CallAfter(self.window.merge_done)
        except:
            err_msg = f''
            wx.CallAfter(self.window.merge_error, err_msg)


def _obstacle_parse_args_from_ui_to_runner(config_from_ui: dict) -> dict:
    runner_config = {
        'WorkDir': os.path.abspath(config_from_ui['WorkFolder']) if config_from_ui['WorkFolder'] != '' else os.path.abspath('.'),
        'TaskName': config_from_ui['TaskName'],
        'Data-Train': (config_from_ui['TrainData'], config_from_ui['TrainDataFormat']),
        'Data-Pred': list(zip(config_from_ui['PredInput'], config_from_ui['PredInputFormat'])),

        'Task-Train': config_from_ui['Task-Train'],
        'Task-Predict': config_from_ui['Task-Predict'],

        'PretrainModel-Ion': config_from_ui['Pretrain-Ion'],
        'PretrainModel-RT': {
            4: config_from_ui['Pretrain-RT-4'],
            5: config_from_ui['Pretrain-RT-5'],
            6: config_from_ui['Pretrain-RT-6'],
            7: config_from_ui['Pretrain-RT-7'],
            8: config_from_ui['Pretrain-RT-8'],
        },
        'ExistedModel-Ion': None,
        'ExistedModel-RT': {l: None for l in [4, 5, 6, 7, 8]},
        'SkipFinetune-Ion': False,
        'SkipFinetune-RT': False,
        'Device': config_from_ui['Device'],
        'Epoch-Ion': int(config_from_ui['Epoch-Ion']),
        'Epoch-RT': int(config_from_ui['Epoch-RT']),
        'BatchSize-Ion': int(config_from_ui['BatchSize-Ion']),
        'BatchSize-RT': int(config_from_ui['BatchSize-RT']),
        'InitLR': float(config_from_ui['InitLR']),
        'MaxPepLen': int(config_from_ui['MaxPepLen']),
        'RTScale': (int(config_from_ui['RTScale-lower']), int(config_from_ui['RTScale-upper'])),
        'EnsembleRT': config_from_ui['EnsembleRT'],

        'NoTime': False,
        'Merge': True,
    }
    return runner_config


class _ObstacleRunnerThread(threading.Thread):
    def __init__(self, window, runner_config, start_time, *args, **kwargs):
        super(_ObstacleRunnerThread, self).__init__(*args, **kwargs)
        self.window = window
        self.runner_config = runner_config
        self.start_time = start_time
        self._running = queue.Queue()

    def terminate(self):
        self._running.put(False)

    def run(self):
        self._running.empty()
        try:
            DeepPhosphoRunner(self.runner_config, start_time=self.start_time, termin_flag=self._running)
            if self._running.qsize() > 0:
                wx.CallAfter(self.window.running_cancel)
            else:
                wx.CallAfter(self.window.running_done, self.runner_config['WorkDir'])
        except Exception:
            tb = traceback.format_exc()
            print(tb)
            wx.CallAfter(self.window.running_error, self.runner_config['WorkDir'])
