import argparse
import os

from deep_phospho import proteomics_utils as prot_utils
from deep_phospho.model_utils.logger import setup_logger

HelpMSG = '''
This script have two usages and both of them build a ready to use spectral library with Spectronaut format

Usage 1: python build_spec_lib.py build -i pathA -r pathB -o pathC
    this will read DeepPhospho predicted ion intensity from pathA and RT from pathB, then generated a library and output to pathC

Usage 2: python build_spec_lib.py merge -l pathA pathB ... -o pathO
    this will merge all passed libraries to one and output to pathO
    [notice] the first library will be used as the major one

For more information, please visit our repository
[DeepPhospho repository] https://github.com/weizhenFrank/DeepPhospho

'''

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter, description=HelpMSG)
    sub_parser = parser.add_subparsers()

    parser_build = sub_parser.add_parser('build')
    parser_build.set_defaults(action='build')
    # ion
    parser_build.add_argument('-i', '--ion', metavar='path', type=str, required=True,
                              help='Path of ion intensity prediction result')
    # rt
    parser_build.add_argument('-r', '--rt', metavar='path', type=str, required=True,
                              help='Path of RT prediction result')
    # output
    parser_build.add_argument('-o', '--output', metavar='path', type=str, required=True,
                              help='Output path of generated library')

    parser_merge = sub_parser.add_parser('merge')
    parser_merge.set_defaults(action='merge')
    # lib
    parser_merge.add_argument('-l', '--lib', metavar='path', type=str, nargs='*', required=True,
                              help='Path of one or more library files')
    # output
    parser_merge.add_argument('-o', '--output', metavar='path', type=str, required=True,
                              help='Output path of merged library')

    args = parser.parse_args()

    if args.action.lower() == 'build':
        logger = setup_logger('Build library', None)
        lib_path = prot_utils.gen_dp_lib.generate_spec_lib(
            data_name='input data',
            output_folder=None,
            pred_ion_path=os.path.abspath(args.ion),
            pred_rt_path=os.path.abspath(args.rt),
            save_path=os.path.abspath(args.output),
            logger=logger
        )
    elif args.action.lower() == 'merge':
        logger = setup_logger('Merge library', None)
        all_libs = args.lib
        if len(all_libs) < 2:
            raise ValueError(f'To merge libraries, the number of input libraries should >= 2')
        path = prot_utils.gen_dp_lib.merge_lib(
            main_lib_path=all_libs[0],
            add_libs_path=dict(enumerate(all_libs[1:], 2)),
            output_folder=None,
            task_name=None,
            save_path=args.output
        )
    else:
        print(HelpMSG)
