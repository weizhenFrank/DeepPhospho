import logging
import os
import random

import numpy as np
import torch

from deep_phospho.model_utils.param_config_load import load_config_from_module, load_config_from_json
from deep_phospho.model_utils.script_arg_parser import choose_config_file, overwrite_config_with_args
from deep_phospho.train_pred_utils.ion_train import train_ion_model

# ---------------- User defined space Start --------------------

"""
Config file can be defined as
    a json file here
    or fill in the config_ion_model.py in DeepPhospho main folder
    or the default config will be used
"""
config_path = r''
SEED = 666

# ---------------- User defined space End --------------------


this_script_dir = os.path.dirname(os.path.abspath(__file__))

config_path, config_dir, config_msg, additional_args = choose_config_file(config_path)

if config_path is not None:
    configs = load_config_from_json(config_path)
else:
    try:
        import config_ion_model as config_module

        config_path = os.path.join(this_script_dir, 'config_ion_model.py')
        config_msg = ('Config file is not in arguments and not defined in script.\n'
                      f'Use config_ion_model.py in DeepPhospho main folder as config file: {config_path}')
    except ModuleNotFoundError:
        from deep_phospho.configs import ion_inten_config as config_module

        config_path = os.path.join(this_script_dir, 'deep_phospho', 'configs', 'ion_inten_config.py')
        config_msg = ('Config file is not in arguments and not defined in script.\n'
                      f'Use default config file ion_inten_config.py in DeepPhospho config module as config file: {config_path}')
    finally:
        configs = load_config_from_module(config_module)
        config_dir = this_script_dir

configs, arg_msg = overwrite_config_with_args(args=additional_args, config=configs)

logging.basicConfig(level=logging.INFO)
torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.backends.cudnn.deterministic = True
torch.autograd.set_detect_anomaly(True)

if __name__ == '__main__':
    train_ion_model(configs=configs, config_load_msgs=config_msg, config_overwrite_msgs=arg_msg)
