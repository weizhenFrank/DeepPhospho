"""
After training IM model on ion intensity model pre-trained weights
Predict IM with trained model
"""

_SEED = 0

Constants = {
    'WorkSpace': './IMWorkSpace',
    'PredData': './IMWorkSpace/Data-IM_Holdout.txt',
    # 'ModPepCol': 'sequence',
    'ModPepCol': 'IntPep',
    # 'PrecChargeCol': 'charge',
    'PrecChargeCol': 'PrecCharge',
    # 'ModPepFormat': 'UniMod',
    'ModPepFormat': 'DP',
    'Device': 'cuda:0',
    'IMModel': './IMWorkSpace/Models/ckpts/best_model.pth',
    'BatchSize': 1024,
    'min_max': (0.5, 1.7),
}

import logging
import os
import random
import sys
from os.path import join as join_path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(-1, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from deep_phospho.model_dataset.preprocess_input_data import Dictionary
from deep_phospho.model_utils.logger import setup_logger
from deep_phospho.model_utils.param_config_load import load_param_from_file
from deep_phospho.proteomics_utils import modpep_format_trans
from im_utils import IMDataset, LSTMTransformerForIM

random.seed(_SEED)
np.random.seed(_SEED)
torch.manual_seed(_SEED)
torch.cuda.manual_seed_all(_SEED)
torch.backends.cudnn.deterministic = True
torch.autograd.set_detect_anomaly(True)

os.makedirs(Constants['WorkSpace'], exist_ok=True)

logging.basicConfig(level=logging.INFO)
logger = setup_logger('IM', Constants['WorkSpace'])

device = torch.device(Constants['Device'])


def preprocess_data_file():
    df = pd.read_csv(Constants['PredData'], sep='\t', low_memory=False)
    df = (df
          .drop_duplicates([Constants['ModPepCol'], Constants['PrecChargeCol']])
          .reset_index(drop=True)
          )
    df['IntPep'] = df[Constants['ModPepCol']].apply({
                                                        'SN13': modpep_format_trans.sn13_to_intpep,
                                                        'MQ1.5': modpep_format_trans.mq1_5_to_intpep,
                                                        'MQ1.6': modpep_format_trans.mq1_6_to_intpep,
                                                        'UniMod': modpep_format_trans.unimodpep_to_intseq,
                                                        'Comet': modpep_format_trans.comet_to_intpep,
                                                        'DP': lambda x: x,
                                                    }[Constants['ModPepFormat']])
    df['PrecCharge'] = df[Constants['PrecChargeCol']]
    df['IntPrec'] = df['IntPep'] + '.' + df['PrecCharge'].astype(str)

    df.to_csv(join_path(Constants['WorkSpace'], 'Data-ProcessedIMPredictionData.txt'), sep='\t', index=False)
    return df


if __name__ == '__main__':
    im_data = preprocess_data_file()

    dictionary = Dictionary()
    if 'X' in dictionary.word2idx:
        dictionary.idx2word.pop(dictionary.word2idx.pop('X'))

    im_dataset = IMDataset(im_data, dictionary, logger, scale_range=Constants['min_max'], name='Prediction', has_im=False)

    im_dataloader = DataLoader(
        dataset=im_dataset,
        batch_size=Constants['BatchSize'],
        shuffle=False,
        num_workers=0,
        collate_fn=None,
        drop_last=False,
    )

    im_model = LSTMTransformerForIM(len(dictionary))
    im_model = load_param_from_file(
        im_model,
        Constants['IMModel'],
        partially=False,
        module_namelist=None,
        logger_name='IM'
    )

    im_model = im_model.to(device)
    im_model.eval()

    logger.info(f'Start prediction')
    pred_ys = []
    with torch.no_grad():
        with tqdm(im_dataloader) as t:
            for iter_idx, (inputs, y) in enumerate(t):
                pred_y = im_model(x1=inputs[0].to(device), x2=inputs[1].to(device))
                pred_ys.append(pred_y.detach().cpu())

        pred_ys = torch.cat(pred_ys).numpy()

    pred_ys = pred_ys * (Constants['min_max'][1] - Constants['min_max'][0]) + Constants['min_max'][0]

    im_data['PredIM'] = pred_ys
    _base_filename = os.path.splitext(os.path.basename(Constants['PredData']))[0]
    im_data.to_csv(join_path(Constants['WorkSpace'], f'Prediction-{_base_filename}.txt'), sep='\t', index=False)
