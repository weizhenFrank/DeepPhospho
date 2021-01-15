import copy
import datetime
import logging
import os
import random
import time

# import ipdb
import ipdb
import numpy as np
import termcolor
import torch
# import ipdb
from deep_phospho.model_dataset.preprocess_input_data import IonData, Dictionary
from deep_phospho.configs import config_main as cfg
from deep_phospho.model_dataset.dataset import IonDataset, collate_fn, RandomMaskingDataset
from deep_phospho.model_utils.ion_eval import eval
from torch.utils.data import DataLoader
from deep_phospho.model_utils.logger import MetricLogger, setup_logger, TFBoardWriter

from deep_phospho.model_utils.lr_scheduler import make_lr_scheduler
from deep_phospho.models.EnsembelModel import LSTMTransformer
from deep_phospho.model_utils.utils_functions import copy_files, get_loss, show_params_status, get_parser
from deep_phospho.model_utils.param_config_load import save_checkpoint, load_param_from_file

logging.basicConfig(level=logging.INFO)
SEED = 666
torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.backends.cudnn.deterministic = True
torch.autograd.set_detect_anomaly(True)
os.environ['CUDA_VISIBLE_DEVICES'] = cfg.TRAINING_HYPER_PARAM['GPU_INDEX']


def main():
    args = get_parser('IonIntensity prediction')
    comment = f'ion_inten-{cfg.data_name}-{cfg.MODEL_CFG["model_name"]}-{args.exp_name}' \
              f'-remove_ac_pep{cfg.TRAINING_HYPER_PARAM["remove_ac_pep"]}' \
              f'-add_phos_principle{cfg.TRAINING_HYPER_PARAM["add_phos_principle"]}' \
              f'-LossType{cfg.TRAINING_HYPER_PARAM["loss_func"]}' \
              f'-use_holdout{args.use_holdout}'

    now = datetime.datetime.now()
    time_str = now.strftime("%Y_%m_%d_%H_%M_%S")
    instance_name = f'{comment}-{time_str}'
    output_dir = os.path.join('../result/ion_inten/', instance_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if args.pretrain_param is not None:
        cfg.TRAINING_HYPER_PARAM['pretrain_param'] = args.pretrain_param

    if args.GPU is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.GPU

    tf_writer_train = TFBoardWriter(output_dir, type='train')
    tf_writer_test = TFBoardWriter(output_dir, type="val")
    tf_writer_holdout = TFBoardWriter(output_dir, type='test')

    logger = setup_logger("IonIntensity", output_dir)

    print("Preparing dataset")

    dictionary = Dictionary(path="../data/20200724-Jeff-MQ_Author-MaxScore_Spec.json")

    Iontrain = IonData(cfg.TRAIN_DATA_CFG, dictionary=dictionary)

    Iontest = IonData(cfg.TEST_DATA_CFG, dictionary=dictionary)

    Ionholdout = IonData(cfg.HOLDOUT_DATA_CFG, dictionary=dictionary)

    if cfg.data_name != 'Prosit':
        if cfg.TRAINING_HYPER_PARAM['Bert_pretrain']:

            train_dataset = RandomMaskingDataset(Iontrain,
                                                 de_mod=True,
                                                 mask_modifier=True,
                                                 mask_ratio=cfg.DATA_PROCESS_CFG['mask_ratio'])
            test_dataset = RandomMaskingDataset(Iontest,
                                                de_mod=True,
                                                mask_modifier=True,
                                                mask_ratio=cfg.DATA_PROCESS_CFG['mask_ratio'])
            holdout_dataset = RandomMaskingDataset(Ionholdout,
                                                   de_mod=True,
                                                   mask_modifier=True,
                                                   mask_ratio=cfg.DATA_PROCESS_CFG['mask_ratio'])

        else:

            train_dataset = IonDataset(Iontrain)
            test_dataset = IonDataset(Iontest)
            holdout_dataset = IonDataset(Ionholdout)

        train_dataloader = DataLoader(dataset=train_dataset,
                                      batch_size=cfg.TRAINING_HYPER_PARAM['BATCH_SIZE'],
                                      shuffle=False,  # to debug, changed to false
                                      num_workers=0,
                                      collate_fn=collate_fn, drop_last=True)

        train_val_dataloader = DataLoader(dataset=train_dataset,
                                          batch_size=256,
                                          shuffle=False,
                                          num_workers=0,
                                          collate_fn=collate_fn)

        test_dataloader = DataLoader(dataset=test_dataset,
                                     shuffle=False,
                                     batch_size=256,
                                     num_workers=0,
                                     collate_fn=collate_fn)
        holdout_dataloader = DataLoader(
            dataset=holdout_dataset,
            batch_size=256,
            shuffle=False,
            num_workers=0,
            collate_fn=collate_fn
        )

    else:

        train_dataset = IonDataset(Iontrain)
        holdout_dataset = IonDataset(Ionholdout)
        index_split = list(range(Iontrain.data_size))
        Train_ratio = 72
        Test_ratio = 18
        train_index = index_split[:int((Train_ratio / (Train_ratio + Test_ratio)) * Iontrain.data_size)]
        test_index = index_split[int((Train_ratio / (Train_ratio + Test_ratio)) * Iontrain.data_size):]
        train_dataloader = DataLoader(dataset=train_dataset,
                                      batch_size=cfg.TRAINING_HYPER_PARAM['BATCH_SIZE'],
                                      sampler=torch.utils.data.SubsetRandomSampler(train_index),
                                      collate_fn=collate_fn)
        train_val_dataloader = DataLoader(dataset=train_dataset,
                                          batch_size=cfg.TRAINING_HYPER_PARAM['BATCH_SIZE'],
                                          sampler=torch.utils.data.SubsetRandomSampler(train_index),
                                          collate_fn=collate_fn)
        test_dataloader = DataLoader(dataset=train_dataset,
                                     batch_size=cfg.TRAINING_HYPER_PARAM['BATCH_SIZE'],
                                     sampler=torch.utils.data.SubsetRandomSampler(test_index),
                                     collate_fn=collate_fn)
        holdout_dataloader = DataLoader(
            dataset=holdout_dataset,
            batch_size=2048,
            shuffle=False,
            collate_fn=collate_fn
        )
    if cfg.TRAINING_HYPER_PARAM['two_stage']:
        loss_func, loss_func_cls = get_loss()
    else:
        loss_func = get_loss()

    # loss_func_eval = copy.deepcopy(loss_func)

    EPOCH = cfg.TRAINING_HYPER_PARAM['EPOCH']
    LR = cfg.TRAINING_HYPER_PARAM['LR']

    if cfg.MODEL_CFG['model_name'] == "LSTMTransformer":
        cfg_to_load = copy.deepcopy(cfg.MODEL_CFG)
        model = LSTMTransformer(
            # ntoken=Iontrain.N_aa,
            RT_mode=cfg.Mode == "RT" or cfg.Mode == "Detect",
            ntoken=31,
            # for prosit, it has 0-21
            use_prosit=cfg.data_name == 'Prosit',
            pdeep2mode=cfg.TRAINING_HYPER_PARAM['pdeep2mode'],
            two_stage=cfg.TRAINING_HYPER_PARAM['two_stage'],
            **cfg_to_load,
        )
    else:
        raise RuntimeError("No valid model name given.")

    logger.info(str(model))
    logger.info("model parameters statuts: \n%s" % show_params_status(model))
    copy_files("deep_phospho/models/ion_model.py", output_dir)
    copy_files("deep_phospho/models/EnsembelModel.py", output_dir)
    copy_files("main_ion.py", output_dir)
    copy_files("deep_phospho/configs", output_dir)

    if cfg.TRAINING_HYPER_PARAM.get("pretrain_param") is not None:
        if cfg.TRAINING_HYPER_PARAM.get("pretrain_param") != '':
            load_param_from_file(model,
                                 cfg.TRAINING_HYPER_PARAM['pretrain_param'],
                                 partially=True,
                                 module_namelist=cfg.TRAINING_HYPER_PARAM['module_namelist'])

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = torch.nn.DataParallel(model)

    model = model.cuda()
    model.train()

    optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad),
                                 LR,
                                 weight_decay=cfg.TRAINING_HYPER_PARAM['weight_decay'])
    scheduler = make_lr_scheduler(optimizer=optimizer, steps=cfg.TRAINING_HYPER_PARAM['LR_STEPS'],
                                  warmup_iters=cfg.TRAINING_HYPER_PARAM['warmup_iters'])

    meters = MetricLogger(delimiter="  ", )
    max_iter = EPOCH * len(train_dataloader)
    start_iter = 0
    start_training_time = time.time()
    end = time.time()
    lambda_cls = cfg.TRAINING_HYPER_PARAM['lambda_cls']
    iteration_best = -1
    best_test_res = -99999999
    best_model = None
    # ipdb.set_trace()

    for epoch in range(EPOCH):
        if cfg.MODEL_CFG['model_name'] == "LSTMTransformer":
            # transform to LSTM + transform end to end finetune mode.
            if epoch >= cfg.TRAINING_HYPER_PARAM['transformer_on_epoch']:
                if not model.transformer_flag:
                    model.set_transformer()
                    logger.info("set transformer on")
                    if cfg.MODEL_CFG['fix_lstm']:
                        fix_eval_modules((model.lstm_list,))

                    # optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), LR*0.5,
                    #                              weight_decay=cfg.TRAINING_HYPER_PARAM['weight_decay'])
                    # scheduler = NoamLR(optimizer,
                    #                    model_size=cfg.MODEL_CFG['lstm_hidden_dim'],
                    #                    warmup_steps=cfg.TRAINING_HYPER_PARAM['warmup_steps'],
                    #                    factor=cfg.TRAINING_HYPER_PARAM['factor'])

        for idx, (inputs, y) in enumerate(train_dataloader):
            # ipdb.set_trace()
            iteration = epoch * len(train_dataloader) + idx
            if len(inputs) == 3:
                # print("3 inputs")
                seq_x = inputs[0]
                x_charge = inputs[1]
                x_nce = inputs[2]
                seq_x = seq_x.cuda()
                x_charge = x_charge.cuda()
                x_nce = x_nce.cuda()
                if cfg.TRAINING_HYPER_PARAM['inter_layer_prediction']:
                    pred_y, inter_out = model(x1=seq_x, x2=x_charge, x3=x_nce)
                else:
                    if cfg.TRAINING_HYPER_PARAM['two_stage']:
                        pred_y, pred_y_cls = model(x1=seq_x, x2=x_charge, x3=x_nce)
                    else:
                        pred_y = model(x1=seq_x, x2=x_charge, x3=x_nce)
            elif len(inputs) == 2:
                seq_x = inputs[0]
                x_charge = inputs[1]
                seq_x = seq_x.cuda()
                # print('-' * 10, seq_x)
                x_charge = x_charge.cuda()
                if cfg.TRAINING_HYPER_PARAM['inter_layer_prediction']:
                    pred_y, inter_out = model(x1=seq_x, x2=x_charge)
                else:
                    if cfg.TRAINING_HYPER_PARAM['two_stage']:
                        pred_y, pred_y_cls = model(x1=seq_x, x2=x_charge)
                    else:
                        pred_y = model(x1=seq_x, x2=x_charge)
            else:
                seq_x = inputs
                seq_x = seq_x.cuda()
                # print('-' * 10, seq_x)
                pred_y = model(x1=seq_x)
            # print('-'*10, x_hydro)
            y = y.cuda()
            # pred_y[torch.where(y == -1)] = -1
            if cfg.TRAINING_HYPER_PARAM['inter_layer_prediction']:
                aux_loss = 0
                for i in range(len(inter_out)):
                    if i < epoch / EPOCH:
                        continue
                    else:
                        aux_loss += loss_func(inter_out[i], y)
                ipdb.set_trace()
                loss = loss_func(pred_y[torch.where(y != -1)], y[torch.where(y != -1)]) + aux_loss

            else:
                if cfg.TRAINING_HYPER_PARAM['two_stage']:
                    # ipdb.set_trace()
                    y_cls = torch.ones_like(y)
                    y_cls[y == -2] = 0
                    y_cls[y == -1] = -1
                    y_no_priori = copy.deepcopy(y)
                    y_no_priori[y == -1] = 0  # padding
                    y_no_priori[y == -2] = 0  # phos
                    y_cls = y_cls.cuda()
                    y_no_priori = y_no_priori.cuda()
                    # ipdb.set_trace()
                    loss_cls = loss_func_cls(pred_y_cls[torch.where(y_cls != -1)], y_cls[torch.where(y_cls != -1)])
                    gate_y = torch.ones_like(y)
                    gate_y[y == -1] = 0
                    gate_y[y == -2] = 0
                    # gate_y[y_no_priori == 0] = 0
                    # this is the bug where we can't assign 0 for prediction where the gt is 0. we only could assign 0 for prediction where gt is -1 or -2
                    try:
                        gated_pred_y = gate_y * pred_y
                    except RuntimeError:
                        ipdb.set_trace()
                    loss_reg = loss_func(gated_pred_y, y_no_priori)
                    loss = lambda_cls * loss_cls + loss_reg

                    meters.update(training_loss_cls=loss_cls.item())
                    meters.update(training_loss_reg=loss_reg.item())
                else:
                    # ipdb.set_trace()
                    # loss = loss_func(pred_y, y)
                    loss = loss_func(pred_y[torch.where(y != -1)], y[torch.where(y != -1)])

            meters.update(training_loss=loss.item())

            # print("----------------")
            # print("pred_y", pred_y, "\t", "y", y)
            # print("----------------")
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

            if iteration % 300 == 0:
                model = model.cuda()
                logger.info(termcolor.colored(meters.delimiter.join(
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
                    instance_name=instance_name,
                    elapsed_time=elapsed_time,
                    iter=idx,
                    EPOCH=EPOCH,
                    epoch=epoch,
                    meters=str(meters),
                    max_iter=len(train_dataloader),
                    total_iter=iteration,
                    lr=optimizer.param_groups[0]["lr"],
                    memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0,
                ), "green")

                )

                lr_rate = optimizer.param_groups[0]['lr']
                tf_writer_train.write_data(iteration / len(train_dataloader), lr_rate, "lr/lr_epoch")
                tf_writer_train.write_data(iteration, lr_rate, "lr/lr_iter")
                # tf_writer_train.write_data(iter='model', meter=[model, *inputs])

            if iteration % cfg.TRAINING_HYPER_PARAM['save_param_interval'] == 0 and iteration != 0 \
                    or idx == len(train_dataloader) - 1:

                # evaluation in training
                if cfg.TRAINING_HYPER_PARAM['Bert_pretrain']:
                    pass
                else:
                    if args.use_holdout:
                        best_test_res, iteration_best, best_model = evaluation(model, logger,
                                                                               tf_writer_train,
                                                                               tf_writer_test,
                                                                               get_loss(),
                                                                               test_dataloader,
                                                                               train_val_dataloader,
                                                                               iteration,
                                                                               best_test_res,
                                                                               iteration_best,
                                                                               best_model,
                                                                               holdout_dataloader=holdout_dataloader,
                                                                               tf_writer_holdout=tf_writer_holdout,
                                                                               use_holdout=False)
                    else:
                        best_test_res, iteration_best, best_model = evaluation(model, logger,
                                                                               tf_writer_train,
                                                                               tf_writer_test,
                                                                               get_loss(),
                                                                               test_dataloader,
                                                                               train_val_dataloader,
                                                                               iteration,
                                                                               best_test_res,
                                                                               iteration_best,
                                                                               best_model,
                                                                               holdout_dataloader=None,
                                                                               tf_writer_holdout=None,
                                                                               use_holdout=False)

                save_checkpoint(model, optimizer, scheduler, output_dir, iteration)
                model.train()
                model = model.cuda()
                torch.cuda.empty_cache()

    save_checkpoint(model, optimizer, scheduler, output_dir, "last_epoch")
    save_checkpoint(best_model, optimizer, scheduler, output_dir, "best_model")
    tf_writer_test.write_data(iteration_best, best_test_res, 'eval_metric/Best_PCC_median')
    logger.info("best_test_res(PCC): %s in iteration %s" % (best_test_res, iteration_best))

    if args.use_holdout:
        model = best_model
        model.cuda()
        evaluation(model, logger, tf_writer_train, tf_writer_test,
                   get_loss(), test_dataloader, train_val_dataloader,
                   iteration=0, best_test_res=None, iteration_best=None,
                   best_model=None, holdout_dataloader=holdout_dataloader,
                   tf_writer_holdout=tf_writer_holdout,
                   use_holdout=True)


def evaluation(model, logger, tf_writer_train, tf_writer_test,
               loss_func_eval, test_dataloader, train_val_dataloader,
               iteration, best_test_res, iteration_best, best_model, holdout_dataloader,
               tf_writer_holdout,
               use_holdout=False):
    model.eval()

    with torch.no_grad():
        if cfg.TRAINING_HYPER_PARAM['Bert_pretrain']:
            pass
        else:
            if not use_holdout:
                logger.info("start evaluation on iteration: %d" % iteration)

                logger.info(termcolor.colored("performance on training set:", "yellow"))
                if cfg.TRAINING_HYPER_PARAM['two_stage']:
                    train_loss, train_reg_loss, train_cls_loss, train_acc, pearson_median, sa_median = eval(model,
                                                                                                            loss_func_eval,
                                                                                                            train_val_dataloader,
                                                                                                            logger,
                                                                                                            iteration)

                    tf_writer_train.write_data(iteration, train_reg_loss, "loss/loss_reg_loss")
                    tf_writer_train.write_data(iteration, train_cls_loss, "loss/loss_cls_loss")
                    tf_writer_train.write_data(iteration, train_acc, 'eval_metric/ion_acc_median')
                else:
                    train_loss, pearson_median, sa_median = eval(model, loss_func_eval, train_val_dataloader, logger,
                                                                 iteration)

                tf_writer_train.write_data(iteration, pearson_median, 'eval_metric/pearson_median')
                tf_writer_train.write_data(iteration, sa_median, 'eval_metric/sa_median')
                tf_writer_train.write_data(iteration, train_loss, "loss/total_loss")

                logger.info(termcolor.colored("performance on validation set:", "yellow"))

                if cfg.TRAINING_HYPER_PARAM['two_stage']:
                    test_loss, test_reg_loss, test_cls_loss, test_acc, pearson_median, sa_median = eval(model,
                                                                                                        loss_func_eval,
                                                                                                        test_dataloader,
                                                                                                        logger,
                                                                                                        iteration)

                    tf_writer_test.write_data(iteration, test_reg_loss, "loss/loss_reg_loss")
                    tf_writer_test.write_data(iteration, test_cls_loss, "loss/loss_cls_loss")
                    tf_writer_test.write_data(iteration, test_acc, 'eval_metric/ion_acc_median')

                else:
                    test_loss, pearson_median, sa_median = eval(model, loss_func_eval, test_dataloader, logger,
                                                                iteration)

                tf_writer_test.write_data(iteration, test_loss, "loss/total_loss")
                tf_writer_test.write_data(iteration, pearson_median, 'eval_metric/pearson_median')
                tf_writer_test.write_data(iteration, sa_median, 'eval_metric/sa_median')

                if pearson_median > best_test_res:
                    best_test_res = pearson_median
                    iteration_best = iteration
                    best_model = copy.deepcopy(model)
                else:
                    best_test_res = best_test_res
                    iteration_best = iteration_best
                    best_model = best_model

                return best_test_res, iteration_best, best_model

            else:
                iteration = 0
                logger.info(termcolor.colored("performance on holdout set:", "yellow"))
                holdout_loss, pearson_median, sa_median = eval(model, loss_func_eval, holdout_dataloader,
                                                               logger, iteration)
                tf_writer_holdout.write_data(iteration, pearson_median, 'test_eval_metric/pearson')
                tf_writer_holdout.write_data(iteration, sa_median, 'test_eval_metric/sa_median')
                tf_writer_holdout.write_data(iteration, holdout_loss, "test/loss")


def fix_eval_modules(eval_modules):
    for module in eval_modules:
        for _, param in module.named_parameters():
            param.requires_grad = False
        # DO NOT use module.eval(), otherwise the module will be in the test mode, i.e., all self.training condition is set to False


if __name__ == '__main__':
    main()
