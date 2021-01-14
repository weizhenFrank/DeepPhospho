from math import sqrt

import ipdb
import numpy as np
import termcolor
import torch
import torch.nn.functional as F
from scipy.stats import pearsonr
from scipy.stats import spearmanr
from torch.utils.data import DataLoader
from tqdm import tqdm
# import ipdb
from deepphospho.utils.utils_functions import Pearson, Delta_t95


def pretrain_eval(model, loss_func, test_dataloader: DataLoader, iteration, logger):
    model.eval()

    loss_log = []
    pred_ys = []
    label_y = []
    hidden_norm = []

    for idx, (seq_x, y) in tqdm(enumerate(test_dataloader),
                                total=len(test_dataloader)):

        seq_x = seq_x.cuda()
        y = y.cuda()
        # print("#" * 10, '\n', y, '\n')
        pred_y = model(seq_x)

        if isinstance(pred_y, tuple):
            pred_y, hidden_vec_norm = pred_y
            pred_y = pred_y.squeeze()

            if hidden_vec_norm is not None:
                hidden_norm.append(hidden_vec_norm.detach().cpu())

        else:
            pred_y = pred_y.squeeze()

        loss = loss_func(pred_y, y)
        # print(loss.item())
        pred_y = F.softmax(pred_y, dim=-1).max(dim=-1)[-1]

        pred_ys.append(pred_y.detach().cpu())
        label_y.append(y.detach().cpu())
        # print('-' * 10, '\n', y.detach().cpu())

        # ipdb.set_trace()

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
    # ipdb.set_trace()

    for each_pred, each_y in zip(pred_ys, label_y):
        # ipdb.set_trace()

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

        all_accs.append(all_acc.mean().item())
        masked_accs.append(mask_acc.mean().item())

    masked_tokens_hit = np.array(masked_tokens_hit).T.sum(axis=-1)
    all_tokens_hit = np.array(all_tokens_hit).T.sum(axis=-1)
    # ipdb.set_trace()

    masked_acc = np.mean(masked_accs)
    all_acc = np.mean(all_accs)
    # ipdb.set_trace()
    logger.info(termcolor.colored("\ntoken prediction accuracy: \n"
                                  f"masked token:  {masked_acc * 100:.3f}:({masked_tokens_hit[0]:10}/{masked_tokens_hit[1]:10})\n" +
                                  f"all token:     {all_acc * 100:.3f}:({all_tokens_hit[0]:10}/{all_tokens_hit[1]:10})\n",
                                  color="green")
                )

    model.train()
    return test_loss, masked_acc, all_acc, hidden_norm


def eval(model, loss_func, test_dataloader: DataLoader, logger, iteration=-1):
    model.eval()
    if hasattr(model, "module"):
        if hasattr(model.module, "transformer_flag"):
            model.module.set_transformer()
    else:
        if hasattr(model, "transformer_flag"):
            if not model.transformer_flag:
                # ipdb.set_trace()
                model.set_transformer()
    logger.info("set transformer on")
    # model.set_transformer()
    logger.info("start validation")
    loss_log = []
    pred_ys = []
    label_y = []
    hidden_norm = []

    for idx, (inputs, y) in tqdm(enumerate(test_dataloader), total=len(test_dataloader)):

        if isinstance(inputs, tuple):
            seq_x, x_hydro, x_rc = inputs
            seq_x = seq_x.cuda()
            x_hydro = x_hydro.cuda()
            x_rc = x_rc.cuda()
            pred_y = model(x1=seq_x, x2=x_hydro, x3=x_rc).squeeze()
        else:
            inputs = inputs.cuda()
            pred_y = model(x1=inputs).squeeze()
        y = y.cuda()

        if isinstance(pred_y, tuple):
            pred_y, hidden_vec_norm = pred_y
            pred_y = pred_y.squeeze()

            if hidden_vec_norm is not None:
                hidden_norm.append(hidden_vec_norm.detach().cpu())

        else:
            pred_y = pred_y.squeeze()

        loss = loss_func(pred_y, y)
        # print(loss.item())

        pred_ys.append(pred_y.detach().cpu())
        label_y.append(y.detach().cpu())
        loss_log.append(loss.item())
    test_loss = np.mean(np.array(loss_log))
    logger.info("\niteration %d, loss %.5f" % (iteration, test_loss))

    pred_ys = torch.cat(pred_ys).numpy()
    label_y = torch.cat(label_y).numpy()
    if len(hidden_norm) != 0:
        hidden_norm = torch.cat(hidden_norm)
    else:
        hidden_norm = None

    pearson_eval = Pearson(label_y, pred_ys)
    delta_t95_eval = Delta_t95(label_y, pred_ys)
    # ipdb.set_trace()
    rt_data = test_dataloader.dataset.ion_data

    # def re_norm(arr):
    #     if rt_data.normalize_by_normal_on:
    #         arr = arr * rt_data.STD_TRAIN + rt_data.MEAN_TRAIN
    #         print("normalize_by_normal_on")
    #     elif rt_data.normalize_by_mean_on:
    #         arr = arr * (rt_data.MAX_RT - rt_data.MIN_RT) + rt_data.MEAN_TRAIN
    #         print("normalize_by_mean_on")
    #     elif rt_data.scale_by_zero_one_on:

    # print("scale_by_zero_one_on")
    def re_norm(arr):
        arr = arr * (rt_data.MAX_RT - rt_data.MIN_RT) + rt_data.MIN_RT
        return arr
    # ipdb.set_trace()
    delta_t95_eval_unnormed = Delta_t95(re_norm(label_y), re_norm(pred_ys))

    logger.info(termcolor.colored("pearson_eval:   %.3f\n" % pearson_eval +
                                  "delta_t95_eval: %.3f\n" % delta_t95_eval +
                                  "                %.3f (un-normed)\n\n" % delta_t95_eval_unnormed, "green"))

    model.train()
    return test_loss, pearson_eval, delta_t95_eval_unnormed, hidden_norm
