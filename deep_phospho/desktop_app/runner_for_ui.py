import threading
from deep_phospho.train_pred_utils.runner import DeepPhosphoRunner


def parse_args_from_ui_to_runner(config_from_ui: dict) -> dict:
    runner_config = {
        'WorkDir': config_from_ui['WorkFolder'] if config_from_ui['WorkFolder'] != '' else '.',
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
    print(runner_config)
    return runner_config


class RunnerThread(threading.Thread):
    def __init__(self, runner_config, start_time, *args, **kwargs):
        super(RunnerThread, self).__init__(*args, **kwargs)
        self.runner_config = runner_config
        self.start_time = start_time
        self._running = True

    def terminate(self):
        self._running = False

    def run(self):
        DeepPhosphoRunner(self.runner_config, start_time=self.start_time)
