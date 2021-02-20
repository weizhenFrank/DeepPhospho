import os
import sys
import time
import datetime
import copy
import logging
import random
from functools import partial

import ipdb
import termcolor

import numpy as np

import torch
from torch.utils.data import DataLoader

from deep_phospho.models.EnsembelModel import LSTMTransformer

from deep_phospho.model_dataset.preprocess_input_data import IonData, Dictionary
from deep_phospho.model_dataset.dataset import IonDataset, collate_fn, RandomMaskingDataset

from deep_phospho.model_utils.ion_eval import eval
from deep_phospho.model_utils.logger import MetricLogger, setup_logger, TFBoardWriter
from deep_phospho.model_utils.lr_scheduler import make_lr_scheduler
from deep_phospho.model_utils.utils_functions import copy_files, get_loss_func, show_params_status, get_parser
from deep_phospho.model_utils.param_config_load import save_checkpoint, load_param_from_file, load_config_as_module, load_config_from_json


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

if config_path != '':
    config_msg = f'Use config file path defined in train script: {config_path}'
elif len(sys.argv) == 2:
    config_path = sys.argv[1]
    config_msg = f'Use config file path defined in command line: {config_path}'
if config_path:
    configs = load_config_from_json(config_path)
    config_dir = os.path.dirname(config_path)
else:
    try:
        import config_ion_model as config_module
        config_path = os.path.join(this_script_dir, 'config_ion_model.py')
        config_msg = f'Use config_ion_model.py in DeepPhospho main folder as config file: {config_path}'
    except ModuleNotFoundError:
        from deep_phospho.configs import ion_inten_config as config_module
        config_path = os.path.join(this_script_dir, 'deep_phospho', 'configs', 'ion_inten_config.py')
        config_msg = f'Use default config file ion_inten_config.py in DeepPhospho config module as config file'
    finally:
        configs = load_config_as_module(config_module)
        config_dir = this_script_dir


logging.basicConfig(level=logging.INFO)
torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.backends.cudnn.deterministic = True
torch.autograd.set_detect_anomaly(True)


def main():

    # Get data path here for ease of use
    train_file = configs['Intensity_DATA_CFG']['TrainPATH']
    test_file = configs['Intensity_DATA_CFG']['TestPATH']
    holdout_file = configs['Intensity_DATA_CFG']['HoldoutPATH']
    if holdout_file:
        use_holdout = True
    else:
        use_holdout = False

    # Define task name as the specific identifier
    task_info = (
        f'ion_inten-{configs["Intensity_DATA_CFG"]["DataName"]}'
        f'-{configs["UsedModelCFG"]["model_name"]}'
        f'-{configs["ExpName"]}'
        f'-remove_ac_pep{configs["TRAINING_HYPER_PARAM"]["remove_ac_pep"]}'
        f'-add_phos_principle{configs["TRAINING_HYPER_PARAM"]["add_phos_principle"]}'
        f'-LossType{configs["TRAINING_HYPER_PARAM"]["loss_func"]}'
        f'-use_holdout{use_holdout}'
    )

    init_time = datetime.datetime.now().strftime("%Y%m%d_%H_%M_%S")
    instance_name = f'{init_time}-{task_info}'

    # Get work folder and define output dir
    work_folder = configs['WorkFolder']
    if work_folder.lower() == 'here' or work_folder == '':
        work_folder = config_dir
    output_dir = os.path.join(work_folder, instance_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Setup logger and add task info and the config msg
    logger = setup_logger("IonInten", output_dir)
    logger.info(f'Work folder is set to {work_folder}')
    logger.info(f'Task start time: {init_time}')
    logger.info(f'Task information: {task_info}')
    logger.info(config_msg)

    # Choose device (Set GPU index or default one, or use CPU)
    if torch.cuda.is_available():
        if configs["TRAINING_HYPER_PARAM"]['GPU_INDEX']:
            device = torch.device(f'cuda:{configs["TRAINING_HYPER_PARAM"]["GPU_INDEX"]}')
            logger.info(f'Cuda available. Use config defined GPU {configs["TRAINING_HYPER_PARAM"]["GPU_INDEX"]}')
        else:
            device = torch.device('cuda:0')
            logger.info(f'Cuda available. No GPU defined in config. Use "cuda:0"')
        use_cuda = True
    else:
        device = torch.device('cpu')
        logger.info(f'Cuda not available. Use CPU')
        use_cuda = False

    # Init tfs
    tf_writer_train = TFBoardWriter(output_dir, type='train')
    tf_writer_test = TFBoardWriter(output_dir, type="val")
    tf_writer_holdout = TFBoardWriter(output_dir, type='test')

    print("Preparing dataset")
    dictionary = Dictionary()
    _ = dictionary.idx2word.pop(dictionary.word2idx.pop('X'))

    ion_train_data = IonData(configs, path=train_file, dictionary=dictionary)
    ion_test_data = IonData(configs, path=test_file, dictionary=dictionary)
    if use_holdout:
        ion_holdout_data = IonData(configs, path=holdout_file, dictionary=dictionary)

    if configs['Intensity_DATA_CFG']['DataName'] != 'Prosit':
        if configs['TRAINING_HYPER_PARAM']['Bert_pretrain']:
            train_dataset = RandomMaskingDataset(ion_train_data,
                                                 de_mod=True,
                                                 mask_modifier=True,
                                                 mask_ratio=configs['Intensity_DATA_PREPROCESS_CFG']['mask_ratio'])
            test_dataset = RandomMaskingDataset(ion_test_data,
                                                de_mod=True,
                                                mask_modifier=True,
                                                mask_ratio=configs['Intensity_DATA_PREPROCESS_CFG']['mask_ratio'])
            if use_holdout:
                holdout_dataset = RandomMaskingDataset(ion_holdout_data,
                                                       de_mod=True,
                                                       mask_modifier=True,
                                                       mask_ratio=configs['Intensity_DATA_PREPROCESS_CFG']['mask_ratio'])

        else:
            train_dataset = IonDataset(ion_train_data, configs)
            test_dataset = IonDataset(ion_test_data, configs)
            if use_holdout:
                holdout_dataset = IonDataset(ion_holdout_data, configs)

        train_dataloader = DataLoader(dataset=train_dataset,
                                      batch_size=configs['TRAINING_HYPER_PARAM']['BATCH_SIZE'],
                                      shuffle=False,  # to debug, changed to false
                                      num_workers=0,
                                      collate_fn=partial(collate_fn, configs=configs), drop_last=True)

        train_val_dataloader = DataLoader(dataset=train_dataset,
                                          batch_size=256,
                                          shuffle=False,
                                          num_workers=0,
                                          collate_fn=partial(collate_fn, configs=configs))

        test_dataloader = DataLoader(dataset=test_dataset,
                                     shuffle=False,
                                     batch_size=256,
                                     num_workers=0,
                                     collate_fn=partial(collate_fn, configs=configs))
        if use_holdout:
            holdout_dataloader = DataLoader(
                dataset=holdout_dataset,
                batch_size=256,
                shuffle=False,
                num_workers=0,
                collate_fn=partial(collate_fn, configs=configs)
            )

    else:
        train_dataset = IonDataset(ion_train_data, configs)
        holdout_dataset = IonDataset(ion_holdout_data, configs)
        index_split = list(range(ion_train_data.data_size))
        Train_ratio = 72
        Test_ratio = 18
        train_index = index_split[:int((Train_ratio / (Train_ratio + Test_ratio)) * ion_train_data.data_size)]
        test_index = index_split[int((Train_ratio / (Train_ratio + Test_ratio)) * ion_train_data.data_size):]
        train_dataloader = DataLoader(dataset=train_dataset,
                                      batch_size=configs.TRAINING_HYPER_PARAM['BATCH_SIZE'],
                                      sampler=torch.utils.data.SubsetRandomSampler(train_index),
                                      collate_fn=partial(collate_fn, configs=configs))
        train_val_dataloader = DataLoader(dataset=train_dataset,
                                          batch_size=configs.TRAINING_HYPER_PARAM['BATCH_SIZE'],
                                          sampler=torch.utils.data.SubsetRandomSampler(train_index),
                                          collate_fn=partial(collate_fn, configs=configs))
        test_dataloader = DataLoader(dataset=train_dataset,
                                     batch_size=configs.TRAINING_HYPER_PARAM['BATCH_SIZE'],
                                     sampler=torch.utils.data.SubsetRandomSampler(test_index),
                                     collate_fn=partial(collate_fn, configs=configs))
        if use_holdout:
            holdout_dataloader = DataLoader(
                dataset=holdout_dataset,
                batch_size=2048,
                shuffle=False,
                collate_fn=partial(collate_fn, configs=configs)
            )
    if configs['TRAINING_HYPER_PARAM']['two_stage']:
        loss_func, loss_func_cls = get_loss_func(configs)
    else:
        loss_func = get_loss_func(configs)

    # loss_func_eval = copy.deepcopy(loss_func)

    EPOCH = configs['TRAINING_HYPER_PARAM']['EPOCH']
    LR = configs['TRAINING_HYPER_PARAM']['LR']

    if configs['UsedModelCFG']['model_name'] == "LSTMTransformer":
        cfg_to_load = copy.deepcopy(configs['UsedModelCFG'])
        model = LSTMTransformer(
            # ntoken=ion_train_data.N_aa,
            RT_mode=False,
            ntoken=len(dictionary),
            # for prosit, it has 0-21
            use_prosit=(configs['Intensity_DATA_CFG']['DataName'] == 'Prosit'),
            pdeep2mode=configs['TRAINING_HYPER_PARAM']['pdeep2mode'],
            two_stage=configs['TRAINING_HYPER_PARAM']['two_stage'],
            **cfg_to_load,
        )
    else:
        raise RuntimeError("No valid model name given.")

    logger.info(str(model))
    logger.info("model parameters statuts: \n%s" % show_params_status(model))

    copy_files(os.path.join(this_script_dir, 'deep_phospho', 'models', 'ion_model.py'), output_dir)
    copy_files(os.path.join(this_script_dir, 'deep_phospho', 'models', 'EnsembelModel.py'), output_dir)
    copy_files(os.path.join(this_script_dir, 'deep_phospho', 'models', 'auxiliary_loss_transformer.py'), output_dir)
    copy_files(os.path.join(this_script_dir, 'train_ion.py'), output_dir)
    copy_files(config_path, output_dir)

    pretrain_param = configs.get('PretrainParam')
    if pretrain_param is not None and pretrain_param != '':
        load_param_from_file(model,
                             pretrain_param,
                             partially=True,
                             module_namelist=configs['TRAINING_HYPER_PARAM']['module_namelist'],
                             logger_name='IonInten')

    model = model.to(device)
    model.train()

    optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad),
                                 LR,
                                 weight_decay=configs['TRAINING_HYPER_PARAM']['weight_decay'])
    scheduler = make_lr_scheduler(optimizer=optimizer, steps=configs['TRAINING_HYPER_PARAM']['LR_STEPS'],
                                  warmup_iters=configs['TRAINING_HYPER_PARAM']['warmup_iters'], configs=configs)

    meters = MetricLogger(delimiter="  ", )
    max_iter = EPOCH * len(train_dataloader)
    start_iter = 0
    start_training_time = time.time()
    end = time.time()
    lambda_cls = configs['TRAINING_HYPER_PARAM']['lambda_cls']
    iteration_best = -1
    best_test_res = -99999999
    best_model = None
    # ipdb.set_trace()

    for epoch in range(EPOCH):
        if configs['UsedModelCFG']['model_name'] == "LSTMTransformer":
            # transform to LSTM + transform end to end finetune mode.
            if epoch >= configs['TRAINING_HYPER_PARAM']['transformer_on_epoch']:
                if not model.transformer_flag:
                    model.set_transformer()
                    logger.info("set transformer on")
                    if configs['UsedModelCFG']['fix_lstm']:
                        fix_eval_modules((model.lstm_list,))

                    # optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), LR*0.5,
                    #                              weight_decay=configs.TRAINING_HYPER_PARAM['weight_decay'])
                    # scheduler = NoamLR(optimizer,
                    #                    model_size=configs['UsedModelCFG']['lstm_hidden_dim'],
                    #                    warmup_steps=configs.TRAINING_HYPER_PARAM['warmup_steps'],
                    #                    factor=configs.TRAINING_HYPER_PARAM['factor'])

        for idx, (inputs, y) in enumerate(train_dataloader):
            # ipdb.set_trace()
            iteration = epoch * len(train_dataloader) + idx
            if len(inputs) == 3:
                # print("3 inputs")
                seq_x = inputs[0]
                x_charge = inputs[1]
                x_nce = inputs[2]
                seq_x = seq_x.to(device)
                x_charge = x_charge.to(device)
                x_nce = x_nce.to(device)
                if configs['TRAINING_HYPER_PARAM']['inter_layer_prediction']:
                    pred_y, inter_out = model(x1=seq_x, x2=x_charge, x3=x_nce)
                else:
                    if configs['TRAINING_HYPER_PARAM']['two_stage']:
                        pred_y, pred_y_cls = model(x1=seq_x, x2=x_charge, x3=x_nce)
                    else:
                        pred_y = model(x1=seq_x, x2=x_charge, x3=x_nce)
            elif len(inputs) == 2:
                seq_x = inputs[0]
                x_charge = inputs[1]
                seq_x = seq_x.to(device)
                # print('-' * 10, seq_x)
                x_charge = x_charge.to(device)
                if configs['TRAINING_HYPER_PARAM']['inter_layer_prediction']:
                    pred_y, inter_out = model(x1=seq_x, x2=x_charge)
                else:
                    if configs['TRAINING_HYPER_PARAM']['two_stage']:
                        pred_y, pred_y_cls = model(x1=seq_x, x2=x_charge)
                    else:
                        pred_y = model(x1=seq_x, x2=x_charge)
            else:
                seq_x = inputs
                seq_x = seq_x.to(device)
                # print('-' * 10, seq_x)
                pred_y = model(x1=seq_x)
            # print('-'*10, x_hydro)
            y = y.to(device)
            # pred_y[torch.where(y == -1)] = -1
            if configs['TRAINING_HYPER_PARAM']['inter_layer_prediction']:
                aux_loss = 0
                for i in range(len(inter_out)):
                    if i < epoch / EPOCH:
                        continue
                    else:
                        aux_loss += loss_func(inter_out[i], y)
                ipdb.set_trace()
                loss = loss_func(pred_y[torch.where(y != -1)], y[torch.where(y != -1)]) + aux_loss

            else:
                if configs['TRAINING_HYPER_PARAM']['two_stage']:
                    # ipdb.set_trace()
                    y_cls = torch.ones_like(y)
                    y_cls[y == -2] = 0
                    y_cls[y == -1] = -1
                    y_no_priori = copy.deepcopy(y)
                    y_no_priori[y == -1] = 0  # padding
                    y_no_priori[y == -2] = 0  # phos
                    y_cls = y_cls.to(device)
                    y_no_priori = y_no_priori.to(device)
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
                model = model.to(device)
                if use_cuda:
                    memory_allo = torch.cuda.max_memory_allocated() / 1024.0 / 1024.0
                else:
                    memory_allo = 0
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
                    memory=memory_allo,
                ), "green")

                )

                lr_rate = optimizer.param_groups[0]['lr']
                tf_writer_train.write_data(iteration / len(train_dataloader), lr_rate, "lr/lr_epoch")
                tf_writer_train.write_data(iteration, lr_rate, "lr/lr_iter")
                # tf_writer_train.write_data(iter='model', meter=[model, *inputs])

            if iteration % configs['TRAINING_HYPER_PARAM']['save_param_interval'] == 0 and iteration != 0 \
                    or idx == len(train_dataloader) - 1:

                # evaluation in training
                if configs['TRAINING_HYPER_PARAM']['Bert_pretrain']:
                    pass
                else:
                    if use_holdout:
                        best_test_res, iteration_best, best_model = evaluation(model, device, logger,
                                                                               tf_writer_train,
                                                                               tf_writer_test,
                                                                               get_loss_func(configs),
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
                        best_test_res, iteration_best, best_model = evaluation(model, device, logger,
                                                                               tf_writer_train,
                                                                               tf_writer_test,
                                                                               get_loss_func(configs),
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
                model = model.to(device)
                if use_cuda:
                    torch.cuda.empty_cache()

    save_checkpoint(model, optimizer, scheduler, output_dir, "last_epoch")
    save_checkpoint(best_model, optimizer, scheduler, output_dir, "best_model")
    tf_writer_test.write_data(iteration_best, best_test_res, 'eval_metric/Best_PCC_median')
    logger.info("best_test_res(PCC): %s in iteration %s" % (best_test_res, iteration_best))

    if use_holdout:
        model = best_model
        model.to(device)
        evaluation(model, device, logger, tf_writer_train, tf_writer_test,
                   get_loss_func(configs), test_dataloader, train_val_dataloader,
                   iteration=0, best_test_res=None, iteration_best=None,
                   best_model=None, holdout_dataloader=holdout_dataloader,
                   tf_writer_holdout=tf_writer_holdout,
                   use_holdout=True)


def evaluation(model, device, logger, tf_writer_train, tf_writer_test,
               loss_func_eval, test_dataloader, train_val_dataloader,
               iteration, best_test_res, iteration_best, best_model, holdout_dataloader,
               tf_writer_holdout,
               use_holdout=False):
    model.eval()

    with torch.no_grad():
        if configs['TRAINING_HYPER_PARAM']['Bert_pretrain']:
            pass
        else:
            if not use_holdout:
                logger.info("start evaluation on iteration: %d" % iteration)

                logger.info(termcolor.colored("performance on training set:", "yellow"))
                if configs['TRAINING_HYPER_PARAM']['two_stage']:
                    train_loss, train_reg_loss, train_cls_loss, train_acc, pearson_median, sa_median = eval(model, configs,
                                                                                                            loss_func_eval,
                                                                                                            train_val_dataloader,
                                                                                                            logger, device=device,
                                                                                                            iteration=iteration)

                    tf_writer_train.write_data(iteration, train_reg_loss, "loss/loss_reg_loss")
                    tf_writer_train.write_data(iteration, train_cls_loss, "loss/loss_cls_loss")
                    tf_writer_train.write_data(iteration, train_acc, 'eval_metric/ion_acc_median')
                else:
                    train_loss, pearson_median, sa_median = eval(model, configs, loss_func_eval, train_val_dataloader, logger,
                                                                 device=device, iteration=iteration)

                tf_writer_train.write_data(iteration, pearson_median, 'eval_metric/pearson_median')
                tf_writer_train.write_data(iteration, sa_median, 'eval_metric/sa_median')
                tf_writer_train.write_data(iteration, train_loss, "loss/total_loss")

                logger.info(termcolor.colored("performance on validation set:", "yellow"))

                if configs['TRAINING_HYPER_PARAM']['two_stage']:
                    test_loss, test_reg_loss, test_cls_loss, test_acc, pearson_median, sa_median = eval(model, configs,
                                                                                                        loss_func_eval,
                                                                                                        test_dataloader,
                                                                                                        logger, device=device,
                                                                                                        iteration=iteration)

                    tf_writer_test.write_data(iteration, test_reg_loss, "loss/loss_reg_loss")
                    tf_writer_test.write_data(iteration, test_cls_loss, "loss/loss_cls_loss")
                    tf_writer_test.write_data(iteration, test_acc, 'eval_metric/ion_acc_median')

                else:
                    test_loss, pearson_median, sa_median = eval(model, configs, loss_func_eval, test_dataloader, logger,
                                                                device=device, iteration=iteration)

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

                if configs['TRAINING_HYPER_PARAM']['two_stage']:
                    holdout_loss, holdout_reg_loss, holdout_cls_loss, holdout_acc, pearson_median, sa_median = eval(
                        model, configs, loss_func_eval, holdout_dataloader, logger, device=device, iteration=iteration)
                else:
                    holdout_loss, pearson_median, sa_median = eval(model, configs, loss_func_eval, holdout_dataloader,
                                                                   logger, device=device, iteration=iteration)

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
