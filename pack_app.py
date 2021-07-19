"""
Packing DeepPhospho desktop APP for:
    - Source code with required python environment
"""

import argparse
import datetime
import os
import shutil
from os.path import join as join_path


def init_arg_parser():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter, description='''''')
    parser.add_argument('-env', '--conda_env', metavar='directory', type=str, default=None, required=True,
                        help='Copy this conda env to DeepPhospho main folder')
    parser.add_argument('-out', '--output_dir', metavar='directory', type=str, default=None, required=False,
                        help='Generate output folder to ...')
    return parser


def recursive_copy(original, target, ignored_items=None, verbose=True, exist_ok=True):
    if ignored_items is None:
        ignored_items = []

    os.makedirs(target, exist_ok=exist_ok)
    curr_items = os.listdir(original)
    for item in curr_items:
        if item in ignored_items:
            continue
        original_item_path = join_path(original, item)
        target_item_path = join_path(target, item)
        if os.path.isdir(original_item_path):
            recursive_copy(original_item_path, target_item_path, ignored_items=ignored_items, verbose=verbose)
        elif os.path.isfile(original_item_path):
            if verbose:
                print(f'copying {item} from {original_item_path} to {target_item_path}')
            shutil.copy(original_item_path, target_item_path)
        else:
            raise
    return 0


def create_app_entry(output_folder):
    with open(join_path(output_folder, 'Launch_DeepPhospho_Desktop.cmd'), 'w') as f:
        f.write('START ./DeepPhosphoPythonEnv/python.exe desktop_app.py')


if __name__ == '__main__':
    arg_parser = init_arg_parser()
    args = arg_parser.parse_args().__dict__

    _this_dir = os.path.abspath('.')

    if args['output_dir'] is None:
        date = datetime.datetime.now().strftime("%Y%m%d")
        OutputDir = join_path(_this_dir, 'release', f'DeepPhospho-{date}-win')
    else:
        OutputDir = os.path.abspath(args['output_dir'])

    ENVFolder = os.path.abspath(args['conda_env'])
    ENVOutputDir = os.path.join(OutputDir, 'DeepPhosphoPythonEnv')

    if os.path.exists(OutputDir):
        print(f'Output dir already existed: {OutputDir}')
        print('Still use this directory to output [y/n]')
        choose = input()
        if choose == 'y':
            pass
        else:
            exit(-1)

    os.makedirs(OutputDir, exist_ok=True)

    ignored_items = [
        '__pycache__',
        '.git',
        '.idea',
        '.gitattributes',
        '.gitignore',
        'DeepPhospho-Data',
        'DeepPhosphoDesktop',
        'Demo-DeepPhosphoRunner',
        'release',
        'PretrainParams',
        'test',
    ]

    print(f'Source code: Copy {_this_dir} to {OutputDir}')
    print(f'Enviroment: Copy {ENVFolder} to {ENVOutputDir}')
    print('Confirm: [y/n]')
    choose = input()
    if choose == 'y':
        recursive_copy(_this_dir, OutputDir, ignored_items=ignored_items, verbose=True, exist_ok=True)
        recursive_copy(ENVFolder, ENVOutputDir, ignored_items=ignored_items, verbose=False, exist_ok=True)
        create_app_entry(OutputDir)
    else:
        exit(-1)
