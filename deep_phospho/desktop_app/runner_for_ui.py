import logging
import os
import threading

import wx
from deep_phospho import proteomics_utils as prot_utils
from deep_phospho.train_pred_utils.runner import DeepPhosphoRunner


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
        'NoTime': False,
        'Merge': True
    }
    return runner_config


class RunnerThread(threading.Thread):
    def __init__(self, window, runner_config, start_time, *args, **kwargs):
        super(RunnerThread, self).__init__(*args, **kwargs)
        self.window = window
        self.runner_config = runner_config
        self.start_time = start_time
        self._running = True

    def run(self):
        try:
            DeepPhosphoRunner(self.runner_config, start_time=self.start_time)
            wx.CallAfter(self.window.running_done, self.runner_config['WorkDir'])
        except:
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
