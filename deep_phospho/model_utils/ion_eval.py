import copy
import logging
from collections import *

import termcolor
from tqdm import tqdm

import numpy as np
from numpy import linalg as LA

from scipy.stats import pearsonr
from scipy.stats import spearmanr

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


def SA(act, pred):
    L2normed_act = act / LA.norm(act)
    L2normed_pred = pred / LA.norm(pred)
    inner_product = np.dot(L2normed_act, L2normed_pred)
    return 1 - 2*np.arccos(inner_product)/np.pi


def Pearson(act, pred):
    return pearsonr(act, pred)[0]


def Spearman(act, pred):
    '''
    Note: there is no need to use spearman correlation for now
    '''
    return spearmanr(act, pred)[0]


def Delta_t95(act, pred):
    num95 = int(np.ceil(len(act) * 0.95))
    return 2 * sorted(abs(act - pred))[num95 - 1]


def Delta_tr95(act, pred):
    return Delta_t95(act, pred) / (max(act) - min(act))


def pretrain_eval(model, loss_func, test_dataloader: DataLoader, iteration, logger):
    model.eval()

    loss_log = []
    pred_ys = []
    label_y = []
    hidden_norm = []

    for idx, (seq_x, x_hydro, y) in tqdm(enumerate(test_dataloader),
                                         total=len(test_dataloader)):

        seq_x = seq_x.cuda()
        # print('-' * 10, seq_x)
        x_hydro = x_hydro.cuda()
        y = y.cuda()
        pred_y = model(seq_x, x_hydro)
        if isinstance(pred_y, tuple):
            pred_y, hidden_vec_norm = pred_y
            pred_y = pred_y.squeeze()

            if hidden_vec_norm is not None:
                hidden_norm.append(hidden_vec_norm.cpu())

        else:
            pred_y = pred_y.squeeze()
        loss = loss_func(pred_y, y)
        # print(loss.item())
        pred_y = F.softmax(pred_y, dim=-1).max(dim=-1)[-1]
        pred_ys.append(pred_y.cpu())
        label_y.append(y.cpu())
        loss_log.append(loss.item())

    test_loss = np.mean(np.array(loss_log))
    logger.info("\niteration %d, loss %.4f" % (iteration, test_loss))

    if len(hidden_norm) != 0:
        hidden_norm = torch.cat(hidden_norm)
    else:
        hidden_norm = None

    masked_accs = []
    all_accs = []
    masked_tokens_hit = []
    all_tokens_hit = []
    for each_pred, each_y in zip(pred_ys, label_y):
        mask = each_y[1]
        label = each_y[0]
        hit_mask = each_pred == label

        hit_mask[label == 0] = 0
        # per sequence predict accuracy

        # all token predict accuracy
        all_acc = hit_mask.float().sum(-1) / (label != 0).sum(-1).float()
        all_tokens_hit.append([hit_mask.float().sum().long().item(),
                               (label != 0).sum().long().item()])
        # masked token predict accuracy
        # take masked tokens hit results and calculate the ( hit num / sequence length)
        mask_acc = (hit_mask.float() * mask.float()).sum(-1) / mask.float().sum(-1)
        masked_tokens_hit.append([(hit_mask.float() * mask.float()).sum().long().item(),
                                  mask.float().sum().long().item()])

        masked_accs.append(mask_acc.mean().item())
        all_accs.append(all_acc.mean().item())

    masked_tokens_hit = np.array(masked_tokens_hit).T.sum(axis=-1)
    all_tokens_hit = np.array(all_tokens_hit).T.sum(axis=-1)

    masked_acc = np.mean(masked_accs)
    all_acc = np.mean(all_accs)
    logger.info(termcolor.colored("\ntoken prediction accuracy: \n"
                                  f"masked token:  {masked_acc * 100:.3f}:({masked_tokens_hit[0]:10}/{masked_tokens_hit[1]:10})\n" +
                                  f"all token:     {all_acc * 100:.3f}:({all_tokens_hit[0]:10}/{all_tokens_hit[1]:10})\n",
                                  color="green")
                )

    model.train()
    return test_loss, masked_acc, hidden_norm


def eval(model, configs, loss_funcs, test_dataloader: DataLoader, logger, iteration=-1):
    model.eval()
    logger = logging.getLogger("IonIntensity")
    logger.info("start validation")
    # ipdb.set_trace()
    loss_log = defaultdict(list)

    pred_ys = []
    label_y = []
    pred_cls_ys = []
    label_cls_ys = []
    hidden_norm = []
    pearson_eval = []
    short_angle = []
    with torch.no_grad():
        for idx, (inputs, y) in tqdm(enumerate(test_dataloader), total=len(test_dataloader)):
            if len(inputs) == 3:
                seq_x = inputs[0]
                x_charge = inputs[1]
                x_nce = inputs[2]
                seq_x = seq_x.cuda()
                x_charge = x_charge.cuda()
                x_nce = x_nce.cuda()
                if configs['TRAINING_HYPER_PARAM']['two_stage']:
                    pred_y, pred_y_cls = model(x1=seq_x, x2=x_charge, x3=x_nce)
                else:
                    pred_y = model(x1=seq_x, x2=x_charge, x3=x_nce)
            else:
                seq_x = inputs[0]
                x_charge = inputs[1]
                seq_x = seq_x.cuda()
                # print('-' * 10, seq_x)
                x_charge = x_charge.cuda()
                if configs['TRAINING_HYPER_PARAM']['two_stage']:
                    pred_y, pred_y_cls = model(x1=seq_x, x2=x_charge)
                else:
                    pred_y = model(x1=seq_x, x2=x_charge)
            y = y.cuda()

            if isinstance(pred_y, tuple):
                pred_y, hidden_vec_norm = pred_y

                if hidden_vec_norm is not None:
                    hidden_norm.append(hidden_vec_norm.cpu())

            # pred_y[torch.where(y == 0)] = 0
            if configs['TRAINING_HYPER_PARAM']['two_stage']:
                loss_func, loss_func_cls = loss_funcs

                lambda_cls = configs['TRAINING_HYPER_PARAM']['lambda_cls']
                y_cls = torch.ones_like(y)
                y_cls[y == -2] = 0   # phos loc is set 0 for cls, nans and value bigger than 0 are set 1 for cls
                y_cls[y == -1] = -1  # padding loc will be ignored for cls
                y_cls = y_cls.cuda()

                loss_cls = loss_func_cls(pred_y_cls[torch.where(y_cls != -1)], y_cls[torch.where(y_cls != -1)])

                y_no_priori = copy.deepcopy(y)
                y_no_priori[y == -1] = 0  # padding
                y_no_priori[y == -2] = 0  # phos
                y_no_priori = y_no_priori.cuda()
                gate_y = torch.ones_like(y)
                gate_y[y_no_priori == 0] = 0
                gated_pred_y = gate_y * pred_y

                loss_reg = loss_func(gated_pred_y, y_no_priori)

                loss = lambda_cls * loss_cls + loss_reg

                loss_log['loss_cls'].append((lambda_cls * loss_cls).item())
                loss_log['loss_reg'].append(loss_reg.item())
                pred_cls_ys.append(pred_y_cls.cpu())
                label_cls_ys.append(y_cls.cpu())

            else:
                pred_y[torch.where(y == -1)] = -1
                pred_y[torch.where(y == -2)] = -2
                loss_func = loss_funcs
                loss = loss_func(pred_y, y)
            loss_log['loss'].append(loss.item())
            # print(loss.item())
            pred_y = pred_y.reshape(pred_y.shape[0], -1)
            y = y.reshape(y.shape[0], -1)
            pred_ys.append(pred_y.cpu())
            label_y.append(y.cpu())

    if configs['TRAINING_HYPER_PARAM']['two_stage']:
        test_loss_cls = np.mean(np.array(loss_log['loss_cls']))
        test_loss_reg = np.mean(np.array(loss_log['loss_reg']))
        logger.info("\niteration %d, loss %.5f" % (iteration, test_loss_cls))
        logger.info("\niteration %d, loss %.5f" % (iteration, test_loss_reg))
        pred_cls_ys = torch.cat(pred_cls_ys)
        label_cls_ys = torch.cat(label_cls_ys)
        acc_stat = defaultdict(list)
        # calculate cls acc for each peptide
        for pred_cls, label_cls in zip(pred_cls_ys, label_cls_ys):
            pred_cls[pred_cls <= 0] = 0
            pred_cls[pred_cls > 0] = 1
            # ipdb.set_trace()
            hit = torch.sum(pred_cls[label_cls != -1] == label_cls[label_cls != -1])
            to_pred = len(label_cls[label_cls != -1])
            acc_pep = hit.item() / to_pred
            acc_stat['hit'].append(hit.item())
            acc_stat['all_to_pred'].append(to_pred)
            acc_stat['acc'].append(acc_pep)
        acc_peps = np.median(np.array(acc_stat['acc']))

    test_loss = np.mean(np.array(loss_log['loss']))
    logger.info("\niteration %d, loss %.5f" % (iteration, test_loss))

    pred_ys = torch.cat(pred_ys).numpy()
    label_y = torch.cat(label_y).numpy()
    # if len(hidden_norm) != 0:
    #     hidden_norm = torch.cat(hidden_norm)
    # else:
    #     hidden_norm = None
    below_cut_counts = 0
    for pred_pep_inten, pep_inten in zip(pred_ys, label_y):
        if len(pep_inten[(pep_inten != 0) * (pep_inten != -1) * (pep_inten != -2)]) < 3:
            # logger.info("pep_inten[(pep_inten != 0) * (pep_inten != -1) * (pep_inten != -2)]) < 3")
            below_cut_counts += 1
            continue
        # ipdb.set_trace()
        select = (pep_inten != 0) * (pep_inten != -1) * (pep_inten != -2)
        pc = Pearson(pred_pep_inten[select], pep_inten[select])
        pearson_eval.append(pc)
        sa = SA(pred_pep_inten[select], pep_inten[select])
        short_angle.append(sa)
    pearson_eval_median = np.median(pearson_eval)
    sa_eval_median = np.median(short_angle)
    if below_cut_counts > 0:
        logger.info(termcolor.colored(f'There is {below_cut_counts} precursors under cut off !', "red"))

    # ipdb.set_trace()

    logger.info(termcolor.colored("pearson_eval:   %.3f\n" % pearson_eval_median))

    model.train()
    if configs['TRAINING_HYPER_PARAM']['two_stage']:
        return test_loss, test_loss_reg, test_loss_cls, acc_peps, pearson_eval_median, sa_eval_median
    else:
        return test_loss, pearson_eval_median, sa_eval_median
