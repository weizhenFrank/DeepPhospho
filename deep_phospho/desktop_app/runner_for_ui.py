import logging
import os
import queue
import threading
import traceback

import wx

from deep_phospho import proteomics_utils as prot_utils
from deep_phospho.train_pred_utils.runner import DeepPhosphoRunner


def check_ui_args(ui_args: dict):
    pass


def parse_args_from_ui_to_runner(config_from_ui: dict) -> dict:
    runner_config = {
        'WorkDir': os.path.abspath(config_from_ui['WorkFolder']) if config_from_ui['WorkFolder'] != '' else os.path.abspath('.'),
        'TaskName': config_from_ui['TaskName'],
        'TrainData': (config_from_ui['TrainData'], config_from_ui['TrainDataFormat']),
        'PredData': list(zip(config_from_ui['PredInput'], config_from_ui['PredInputFormat'])),
        'ExistedIonModel': None,
        'ExistedRTModel': None,
        'SkipIonFinetune': False,
        'SkipRTFinetune': False,
        'Device': config_from_ui['Device'],
        'IonEpoch': int(config_from_ui['Epoch-Ion']),
        'RTEpoch': int(config_from_ui['Epoch-RT']),
        'IonBatchSize': int(config_from_ui['BatchSize-Ion']),
        'RTBatchSize': int(config_from_ui['BatchSize-RT']),
        'InitLR': float(config_from_ui['InitLR']),
        'MaxPepLen': int(config_from_ui['MaxPepLen']),
        'RTScale': (int(config_from_ui['RTScale-lower']), int(config_from_ui['RTScale-upper'])),
        'EnsembleRT': config_from_ui['RTEnsemble'],
        # 'TrainMode': config_from_ui['TrainMode'],
        'Pretrain-Ion': config_from_ui['Pretrain-Ion'],
        'Pretrain-RT-4': config_from_ui['Pretrain-RT-4'],
        'Pretrain-RT-5': config_from_ui['Pretrain-RT-5'],
        'Pretrain-RT-6': config_from_ui['Pretrain-RT-6'],
        'Pretrain-RT-7': config_from_ui['Pretrain-RT-7'],
        'Pretrain-RT-8': config_from_ui['Pretrain-RT-8'],
        'NoTime': False,
        'Merge': True,
        'Task-Train': config_from_ui['Task-Train'],
        'Task-Predict': config_from_ui['Task-Predict'],
    }
    return runner_config


class RunnerThread(threading.Thread):
    def __init__(self, window, runner_config, start_time, *args, **kwargs):
        super(RunnerThread, self).__init__(*args, **kwargs)
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
            wx.CallAfter(self.window.running_done, self.runner_config['WorkDir'])
        except Exception:
            tb = traceback.format_exc()
            print(tb)
            wx.CallAfter(self.window.running_error, self.runner_config['WorkDir'])


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
