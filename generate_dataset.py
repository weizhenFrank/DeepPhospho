import argparse

from deep_phospho import proteomics_utils as prot_utils

HelpMSG = '''
This script have two usages to transform data to DeepPhospho training or prediction data

Usage 1: python generate_dataset.py train -f pathA -t file_type -o dirA
    this will transform the input train file with format (file_type) to DeepPhospho training data format and output to directory dirA

Usage 2: python generate_dataset.py pred -f pathA -t file_type -o dirA
    this will transform the input prediction file with format (file_type) to DeepPhospho prediction input format and output to directory dirA

For more information, please visit our repository
[DeepPhospho repository] https://github.com/weizhenFrank/DeepPhospho

'''

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter, description=HelpMSG)
    sub_parser = parser.add_subparsers()

    parser_train = sub_parser.add_parser('train')
    parser_train.set_defaults(action='train')
    # file
    parser_train.add_argument('-f', '--file', metavar='path', type=str, required=True,
                              help='Path of training data')
    # type
    parser_train.add_argument('-t', '--type', metavar='str', type=str, required=True,
                              help='''The format (or source) of input file, and the valid values are listed as below:
    SNLib - Spectronaut library
    MQ1.5 - MaxQuant msms.txt with version <= 1.5
    MQ1.6 - MaxQuant msms.txt with version >= 1.6''')
    # output
    parser_train.add_argument('-o', '--output', metavar='dir', type=str, required=True,
                              help='Directory to store the generated data')

    parser_pred = sub_parser.add_parser('pred')
    parser_pred.set_defaults(action='pred')
    # file
    parser_pred.add_argument('-f', '--file', metavar='path', type=str, required=True,
                             help='Path of prediction data')
    # type
    parser_pred.add_argument('-t', '--type', metavar='str', type=str, required=True,
                             help='''The prediction data can have multi format:
I. "SNLib" for Spectronaut library
II. "SNResult" for Spectronaut result
III. "MQ1.5" or "MQ1.6" for msms.txt/evidence.txt from MaxQuant version <= 1.5 or >= 1.6
IV. any tab-separated file with two columns "sequence" and "charge. And for "sequence" column, the modified peptides in the following format are valid
    a. "PepSN13" is Spectronaut 13+ peptide format like _[Acetyl (Protein N-term)]M[Oxidation (M)]LSLS[Phospho (STY)]PLK_
    b. "PepMQ1.5" is MaxQuant 1.5- peptide format like _(ac)GS(ph)QDM(ox)GS(ph)PLRET(ph)RK_
    c. "PepMQ1.6" is MaxQuant 1.6+ peptide format like _(Acetyl (Protein N-term))TM(Oxidation (M))DKS(Phospho (STY))ELVQK_
    d. "PepComet" is Comet peptide format like n#DFM*SPKFS@LT@DVEY@PAWCQDDEVPITM*QEIR
    e. "PepDP" is DeepPhospho used peptide format like *1ED2MCLK''')
    # output
    parser_pred.add_argument('-o', '--output', metavar='dir', type=str, required=True,
                             help='Directory to store the generated data')

    args = parser.parse_args()

    if args.action.lower() == 'train':
        prot_utils.dp_train_data.file_to_trainset(path=args.file, output_folder=args.output, file_type=args.type)
    elif args.action.lower() == 'pred':
        prot_utils.dp_pred_data.file_to_pred_input(path=args.file, output_folder=args.output, file_type=args.type)
    else:
        print(HelpMSG)
