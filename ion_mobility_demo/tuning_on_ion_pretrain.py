"""
Transfer learning for ion mobility
Will use weights of peptide embedding, charge embedding, lstm layers, Transformer layers, from pretrained ion intensity model
A FC layer will be newly generated and initialized
Can choose to fix layers except the last FC or tune all parameters
"""

_SEED = 0

HyperParams = {
    'epoch': 10,
    'lr': 0.0001,
    'weight_decay': 1e-8,
    'train_batch_size': 256,
    'test_batch_size': 1024,
    'min_max': (0.5, 1.7),
    'fix_layers_except_fc': True,
}

Constants = {
    'WorkSpace': './IMWorkSpace',
    'TrainingData': r'',
    # 'ModPepCol': 'ModifiedPeptideSequence',
    'ModPepCol': 'ModifiedPeptide',
    'PrecChargeCol': 'PrecursorCharge',
    # 'IMCol': 'PrecursorIonMobility',
    'IMCol': 'IonMobility',
    # 'ModPepFormat': 'UniMod',
    'ModPepFormat': 'SN13',
    'DataSplitRatio': (70, 18, 12),
    'Device': 'cuda:0',
    'PretrainIonModel': '../PretrainParams/IonModel/best_model.pth',
    'RecordInterval': 300,
    'SaveModelInterval': 1200,
}

import copy
import datetime
import logging
import os
import random
import sys
import time
from os.path import join as join_path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(-1, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from deep_phospho.model_dataset.preprocess_input_data import Dictionary
from deep_phospho.model_utils.logger import MetricLogger
from deep_phospho.model_utils.logger import setup_logger
from deep_phospho.model_utils.param_config_load import load_weight_partially_from_file
from deep_phospho.model_utils.param_config_load import save_checkpoint
from deep_phospho.model_utils.utils_functions import Pearson, Delta_t95, RMSELoss
from deep_phospho.model_utils.lr_scheduler import WarmupMultiStepLR
from deep_phospho.proteomics_utils import modpep_format_trans
from deep_phospho.proteomics_utils.dp_train_data import split_nparts
from im_utils import IMDataset, LSTMTransformerForIM

random.seed(_SEED)
np.random.seed(_SEED)
torch.manual_seed(_SEED)
torch.cuda.manual_seed_all(_SEED)
torch.backends.cudnn.deterministic = True
torch.autograd.set_detect_anomaly(True)

os.makedirs(Constants['WorkSpace'], exist_ok=True)
model_save_folder = join_path(Constants['WorkSpace'], 'Models')
os.makedirs(model_save_folder, exist_ok=True)

logging.basicConfig(level=logging.INFO)
logger = setup_logger('IM', Constants['WorkSpace'])

device = torch.device(Constants['Device'])


def preprocess_data_file():
    df = pd.read_csv(Constants['TrainingData'], sep='\t', low_memory=False)
    df = (df
          .drop_duplicates([Constants['ModPepCol'], Constants['PrecChargeCol']])
          .reset_index(drop=True)
          .rename(columns={Constants['PrecChargeCol']: 'PrecCharge', Constants['IMCol']: 'IM'})
          )
    df['IntPep'] = df[Constants['ModPepCol']].apply({
                                                        'SN13': modpep_format_trans.sn13_to_intpep,
                                                        'MQ1.5': modpep_format_trans.mq1_5_to_intpep,
                                                        'MQ1.6': modpep_format_trans.mq1_6_to_intpep,
                                                        'UniMod': modpep_format_trans.unimodpep_to_intseq,
                                                        'Comet': modpep_format_trans.comet_to_intpep,
                                                        'DP': lambda x: x,
                                                    }[Constants['ModPepFormat']])
    df['IntPrec'] = df['IntPep'] + '.' + df['PrecCharge'].astype(str)

    df = df[['IntPep', 'PrecCharge', 'IntPrec', 'IM']]
    df.to_csv(join_path(Constants['WorkSpace'], 'Data-ProcessedIMTrainingData.txt'), sep='\t', index=False)

    _im_data = dict()
    for idx, split_prec in enumerate(split_nparts(df['IntPrec'].tolist(), ratios=Constants['DataSplitRatio'], seed=_SEED)):
        data_type = ['Train', 'Val', 'Holdout'][idx]

        _im_data[data_type] = df[df['IntPrec'].isin(split_prec)].copy()
        logger.info(f'Precursors in {data_type}: {len(_im_data[data_type])}')

        _im_data[data_type].to_csv(join_path(Constants['WorkSpace'], f'Data-IM_{data_type}.txt'), sep='\t', index=False)

    return _im_data


def evaluate(
        model, test_dataloader,
        loss_func,
        logger, device,
        iteration=-1
):
    model.eval()

    logger.info("Start validation")
    loss_log = []
    pred_ys = []
    label_y = []

    with tqdm(enumerate(test_dataloader), total=len(test_dataloader)) as t:
        for idx, (inputs, y) in t:
            pred_y = model(x1=inputs[0].to(device), x2=inputs[1].to(device)).squeeze()
            y = y.to(device).squeeze()

            loss = loss_func(pred_y, y)

            pred_ys.append(pred_y.detach().cpu())
            label_y.append(y.detach().cpu())
            loss_log.append(loss.item())
    test_loss = np.mean(np.array(loss_log))
    logger.info("\niteration %d, loss %.5f" % (iteration, test_loss))

    pred_ys = torch.cat(pred_ys).numpy().reshape(-1)
    label_y = torch.cat(label_y).numpy().reshape(-1)

    pearson_eval = Pearson(label_y, pred_ys)
    delta_t95_eval = Delta_t95(label_y, pred_ys)

    logger.info("pearson_eval:   %.3f\n" % pearson_eval +
                "delta_t95_eval: %.3f\n" % delta_t95_eval)

    model.train()
    return test_loss, pearson_eval, delta_t95_eval


def perform_evaluation(
        model,
        loss_func_eval,
        train_val_dataloader, test_dataloader,
        iteration, best_test_res, iteration_best, best_model,
        device, logger,
        holdout_dataloader=None,
):
    model.eval()

    with torch.no_grad():
        logger.info("start evaluation on iteration: %d" % iteration)
        logger.info("performance on training set:")
        training_loss, pearson, delta_t95 = evaluate(
            model, train_val_dataloader, loss_func_eval,
            logger, device=device, iteration=iteration
        )

        logger.info("performance on validation set:")
        test_loss, pearson, delta_t95 = evaluate(
            model, test_dataloader, loss_func_eval,
            logger, device=device, iteration=iteration)
        if delta_t95 < best_test_res:
            best_test_res = delta_t95
            iteration_best = iteration
            best_model = copy.deepcopy(model)
        else:
            best_test_res = best_test_res
            iteration_best = iteration_best
            best_model = best_model

        if holdout_dataloader is not None:
            iteration = 0
            logger.info("performance on holdout set:")
            holdout_loss, pearson, delta_t95 = evaluate(
                model, holdout_dataloader, loss_func_eval,
                logger, device=device, iteration=iteration)

        return best_test_res, iteration_best, best_model


if __name__ == '__main__':
    im_data = preprocess_data_file()

    dictionary = Dictionary()
    if 'X' in dictionary.word2idx:
        dictionary.idx2word.pop(dictionary.word2idx.pop('X'))

    im_datasets = dict()
    for name, df in im_data.items():
        im_datasets[name] = IMDataset(df, dictionary, logger, scale_range=HyperParams['min_max'], name=name, has_im=True)

    im_dataloaders = dict()
    for name, dataset in im_datasets.items():
        im_dataloaders[name] = DataLoader(
            dataset=dataset,
            batch_size=HyperParams['train_batch_size'] if name == 'Train' else HyperParams['test_batch_size'],
            shuffle=False,
            num_workers=0,
            collate_fn=None,
            drop_last=True if name == 'Train' else False,
        )
    im_dataloaders['TrainEval'] = copy.deepcopy(im_dataloaders['Train'])

    im_model = LSTMTransformerForIM(len(dictionary))
    im_model = load_weight_partially_from_file(
        im_model,
        Constants['PretrainIonModel'],
        module_namelist=None,
        logger_name='IM',
        verbose_only_miss_matched=True
    )

    # im_model = CNN(len(dictionary))

    if HyperParams['fix_layers_except_fc']:
        for name, module in im_model.named_modules():
            if not name:
                continue
            if any([(name in _ or _ in name) for _ in ('weight_layer', 'output_linear')]):
                logger.info(f'Will train module {name}')
            else:
                logger.info(f'Fix module {name}')
                for _, param in module.named_parameters():
                    param.requires_grad = False

    im_model = im_model.to(device)
    im_model.train()

    optimizer = torch.optim.Adam(
        (p for p in im_model.parameters() if p.requires_grad),
        HyperParams['lr'],
        weight_decay=HyperParams['weight_decay']
    )
    scheduler = WarmupMultiStepLR(
        optimizer,
        milestones=[10000, 20000],
        warmup_iters=0,
        gamma=0.1,
        warmup_factor=1.0 / 3,
        warmup_method="linear",
    )
    loss_func = RMSELoss()

    logger.info('Check param with grad required')
    for name, param in im_model.named_parameters():
        logger.info(name)
        logger.info(param.requires_grad)

    meters = MetricLogger(delimiter="  ", )

    max_iter = HyperParams['epoch'] * len(im_dataloaders['Train'])
    start_iter = 0
    iteration_best = -1
    best_test_res = -99999999
    best_model = None

    start_training_time = time.time()
    end = time.time()

    for epoch in range(HyperParams['epoch']):
        for iter_idx, (inputs, y) in enumerate(im_dataloaders['Train']):

            iteration = epoch * len(im_dataloaders['Train']) + iter_idx

            pred_y = im_model(x1=inputs[0].to(device), x2=inputs[1].to(device))
            y = y.to(device)

            loss = loss_func(pred_y.squeeze(), y.squeeze())
            meters.update(training_loss=loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            batch_time = time.time() - end
            data_time = time.time() - end
            end = time.time()
            meters.update(time=batch_time, data=data_time)

            eta_seconds = meters.time.global_avg * (max_iter - iteration)
            eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
            elapsed_time = str(datetime.timedelta(seconds=int(end - start_training_time)))

            if iteration % Constants['RecordInterval'] == 0:
                if torch.cuda.is_available():
                    memory_allo = torch.cuda.max_memory_allocated() / 1024.0 / 1024.0
                else:
                    memory_allo = 0
                logger.info(meters.delimiter.join(
                    [
                        "\ninstance id: {instance_name}\n",
                        "elapsed time: {elapsed_time}\n",
                        "eta: {eta}\n",
                        "iter: {iter}/{max_iter} -- {total_iter}\n",
                        "epoch: {epoch}/{EPOCH}\n"
                        "  {meters}",
                        "lr: {lr:.6f}\n",
                        "max mem: {memory:.0f}\n",
                    ]
                ).format(
                    eta=eta_string,
                    instance_name=None,
                    elapsed_time=elapsed_time,
                    iter=iter_idx,
                    EPOCH=HyperParams['epoch'],
                    epoch=epoch + 1,
                    meters=str(meters),
                    max_iter=len(im_dataloaders['Train']),
                    total_iter=iteration,
                    lr=optimizer.param_groups[0]["lr"],
                    memory=memory_allo,
                ))

            if (iteration % Constants['SaveModelInterval'] == 0 and iteration != 0) or (iter_idx == len(im_dataloaders['Train']) - 2):
                best_test_res, iteration_best, best_model = perform_evaluation(
                    im_model,
                    loss_func,
                    im_dataloaders['TrainEval'], im_dataloaders['Val'],
                    iteration, best_test_res, iteration_best, best_model,
                    device, logger,
                    holdout_dataloader=None
                )
                save_checkpoint(im_model, optimizer, scheduler, model_save_folder, iteration)
                im_model.train()
                im_model = im_model.to(device)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    if best_model is None:
        best_model = copy.deepcopy(im_model)
    save_checkpoint(im_model, optimizer, scheduler, model_save_folder, "last_epochless")
    save_checkpoint(best_model, optimizer, scheduler, model_save_folder, "best_model")

    im_model = best_model
    im_model.to(device)
    _ = perform_evaluation(
        im_model,
        loss_func,
        im_dataloaders['TrainEval'], im_dataloaders['Val'],
        iteration, best_test_res, iteration_best, best_model,
        device, logger,
        holdout_dataloader=im_dataloaders['Holdout']
    )
