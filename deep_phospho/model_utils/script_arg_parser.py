import os
import sys
import argparse


def init_argparser():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter, description='''
    ''')
    # config file
    parser.add_argument('-c', '--config', metavar='Path', type=str, default=None,
                        help='''Config file with json format. The templete is avaliable in the main folder of DeepPhospho.''')
    # gpu idx
    parser.add_argument('-g', '--gpuidx', metavar='GPU idx or CPU', default=None,
                        help='''Set the used GPU idx (0, 1, 2, ...) or "cpu" to use cpu only. Default: 0''')
    # exp name
    parser.add_argument('-e', '--expname', metavar='str', type=str, default='',
                        help='Name of the task under execution')
    # dataset name
    parser.add_argument('-d', '--dataname', metavar='str', type=str, default='',
                        help='Name of the used dataset')
    # encoder layer
    parser.add_argument('-l', '--layernum', metavar='int', type=int, default=None,
                        help='Number of the transformer encoder layers')
    return parser


def parse_args() -> dict:
    arg_parser = init_argparser()
    args = arg_parser.parse_args()
    return args.__dict__


def detect_arg_type() -> (str, dict):
    all_args = sys.argv
    if len(all_args) == 2 and all_args[1] not in ('-h', '--help'):
        arg_config_path = all_args[1]
        args = {}
    else:
        args = {k: v for k, v in parse_args().items() if v is not None and v != '' and not k.startswith('_')}
        arg_config_path = args.get('config')
    return arg_config_path, args


def choose_config_file(config_path_in_script: str):
    """
    :param config_path_in_script:
    """
    arg_config_path, args = detect_arg_type()

    if arg_config_path is not None and arg_config_path != '':
        config_file = arg_config_path
        config_dir = os.path.dirname(config_file)
        config_msg = ('Config file is passed from arguments, and is not None or empty string.\n'
                      f'Use config file: {config_file}')
    elif config_path_in_script is not None and config_path_in_script != '':
        config_file = config_path_in_script
        config_dir = os.path.dirname(config_file)
        config_msg = ('Config file is defined in script, and not in arguments.\n'
                      f'Use config file: {config_file}')
    else:
        config_file = None
        config_dir = None
        config_msg = None
    return config_file, config_dir, config_msg, args


def overwrite_config_with_args(args: dict, config: dict, logger=None) -> (dict, str):
    if args:
        joined_args = ''
        for k, v in args.items():
            if k == 'gpuidx':
                config['TRAINING_HYPER_PARAM']['GPU_INDEX'] = v
            elif k == 'expname':
                config['ExpName'] = v
            elif k == 'dataname':
                data_field_name = 'Intensity_DATA_CFG' if 'Intensity_DATA_CFG' in config else 'RT_DATA_CFG'
                config[data_field_name]['DataName'] = v
            elif k == 'layernum':
                config['UsedModelCFG']['num_encd_layer'] = v
            else:
                pass
            joined_args += f'\t{k}: {v}'
        msg = f'Following configs are overwriten by the passed arguments:\n{joined_args}'
    else:
        msg = 'No argument is passed'
    return config, msg

