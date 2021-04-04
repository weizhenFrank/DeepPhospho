import copy
import datetime
import logging
import os
import random
import time
from functools import partial

import dill
import termcolor

import numpy as np
import torch
from torch.utils.data import DataLoader

from deep_phospho.model_dataset.dataset import IonDataset, collate_fn
from deep_phospho.model_dataset.preprocess_input_data import RTdata, Dictionary
from deep_phospho.model_utils.logger import MetricLogger, setup_logger, save_config, TFBoardWriter
from deep_phospho.model_utils.lr_scheduler import make_lr_scheduler
from deep_phospho.model_utils.param_config_load import save_checkpoint, load_param_from_file, load_config_as_module, load_config_from_json
from deep_phospho.model_utils.rt_eval import eval
from deep_phospho.model_utils.script_arg_parser import choose_config_file, overwrite_config_with_args
from deep_phospho.model_utils.utils_functions import copy_files, get_loss_func, show_params_status
from deep_phospho.models.EnsembelModel import LSTMTransformer

# ---------------- User defined space Start --------------------

"""
Config file can be defined as
    a json file here
    or fill in the config_rt_model.py in DeepPhospho main folder
    or the default config will be used
"""
config_path = r''
SEED = 666

# ---------------- User defined space End --------------------


this_script_dir = os.path.dirname(os.path.abspath(__file__))

config_path, config_dir, config_msg, additional_args = choose_config_file(config_path)

if config_path is not None:
    configs = load_config_from_json(config_path)
else:
    try:
        import config_rt_model as config_module

        config_path = os.path.join(this_script_dir, 'config_rt_model.py')
        config_msg = ('Config file is not in arguments and not defined in script.\n'
                      f'Use config_rt_model.py in DeepPhospho main folder as config file: {config_path}')
    except ModuleNotFoundError:
        from deep_phospho.configs import rt_config as config_module

        config_path = os.path.join(this_script_dir, 'deep_phospho', 'configs', 'rt_config.py')
        config_msg = ('Config file is not in arguments and not defined in script.\n'
                      f'Use default config file rt_config.py in DeepPhospho config module as config file: {config_path}')
    finally:
        configs = load_config_as_module(config_module)
        config_dir = this_script_dir

configs, arg_msg = overwrite_config_with_args(args=additional_args, config=configs)

logging.basicConfig(level=logging.INFO)
torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.backends.cudnn.deterministic = True  # To test dilated conv, the result shows set true make conv dilation slower 7X
# change as True to test the Jeff result huge difference
torch.autograd.set_detect_anomaly(True)


def main():
    # Get data path here for ease of use
    train_file = configs['RT_DATA_CFG']['TrainPATH']
    test_file = configs['RT_DATA_CFG']['TestPATH']
    holdout_file = configs['RT_DATA_CFG']['HoldoutPATH']
    if holdout_file:
        use_holdout = True
    else:
        use_holdout = False

    # Define task name as the specific identifier
    resume = '-RESUME' if configs['TRAINING_HYPER_PARAM']['resume'] else ''
    task_info = (
        f'{configs["RT_DATA_CFG"]["DataName"]}'
        f'-{configs["UsedModelCFG"]["model_name"]}'
        f'-{configs["ExpName"]}'
        f'-EncdNum{configs["UsedModelCFG"]["num_encd_layer"]}'
        f'{resume}'
    )
    init_time = datetime.datetime.now().strftime("%Y%m%d_%H_%M_%S")

    if configs['InstanceName'] != '':
        instance_name = configs['InstanceName']
        instance_name_msg = f'Use manurally defined instance name {instance_name}'
    else:
        instance_name = f'{init_time}-{task_info}'
        instance_name_msg = f'No instance name defined in config or passed from arguments. Use {instance_name}'

    # Get work folder and define output dir
    work_folder = configs['WorkFolder']
    if work_folder.lower() == 'here' or work_folder == '':
        work_folder = config_dir
    output_dir = os.path.join(work_folder, instance_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Setup logger and add task info and the config msg
    logger = setup_logger("RT", output_dir)
    logger.info(f'Work folder is set to {work_folder}')
    logger.info(f'Task start time: {init_time}')
    logger.info(f'Task information: {task_info}')
    logger.info(f'Instance name: {instance_name_msg}')
    logger.info(arg_msg)
    logger.info(config_msg)
    logger.info(save_config(configs, output_dir))

    # Choose device (Set GPU index or default one, or use CPU)
    if configs["TRAINING_HYPER_PARAM"]['GPU_INDEX'].lower() == 'cpu':
        device = torch.device('cpu')
        logger.info(f'CPU is defined as the device in config')
    elif torch.cuda.is_available():
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

    rt_train_data = RTdata(configs, train_file, dictionary=dictionary)
    rt_test_data = RTdata(configs, test_file, dictionary=dictionary)

    train_dataset = IonDataset(rt_train_data, configs)
    test_dataset = IonDataset(rt_test_data, configs)

    train_dataloader = DataLoader(dataset=train_dataset,
                                  batch_size=configs['TRAINING_HYPER_PARAM']['BATCH_SIZE'],
                                  shuffle=False,
                                  num_workers=4,
                                  collate_fn=partial(collate_fn, configs=configs), drop_last=True)

    train_val_dataloader = DataLoader(dataset=train_dataset,
                                      batch_size=1024,
                                      shuffle=False,
                                      num_workers=2,
                                      collate_fn=partial(collate_fn, configs=configs))

    test_dataloader = DataLoader(dataset=test_dataset,
                                 shuffle=False,
                                 batch_size=1024,
                                 num_workers=2,
                                 collate_fn=partial(collate_fn, configs=configs))

    if use_holdout:
        rt_holdout_data = RTdata(configs, holdout_file, dictionary=dictionary)
        holdout_dataset = IonDataset(rt_holdout_data, configs)
        holdout_dataloader = DataLoader(
            dataset=holdout_dataset,
            batch_size=256,
            shuffle=False,
            num_workers=2,
            collate_fn=partial(collate_fn, configs=configs)
        )

    loss_func = get_loss_func(configs)
    loss_func_eval = copy.deepcopy(loss_func)

    EPOCH = configs['TRAINING_HYPER_PARAM']['EPOCH']
    LR = configs['TRAINING_HYPER_PARAM']['LR']

    if configs['UsedModelCFG']['model_name'] == "LSTMTransformer":
        cfg_to_load = copy.deepcopy(configs['UsedModelCFG'])

        model = LSTMTransformer(
            # ntoken=Iontrain.N_aa,
            RT_mode=True,
            ntoken=rt_train_data.N_aa,
            # for prosit, it has 0-21
            **cfg_to_load,
        )

    else:
        raise RuntimeError("No valid model name given.")

    logger.info(str(model))
    logger.info("model parameters statuts: \n%s" % show_params_status(model))

    copy_files(os.path.join(this_script_dir, 'deep_phospho', 'models', 'ion_model.py'), output_dir)
    copy_files(os.path.join(this_script_dir, 'deep_phospho', 'models', 'EnsembelModel.py'), output_dir)
    copy_files(os.path.join(this_script_dir, 'deep_phospho', 'models', 'auxiliary_loss_transformer.py'), output_dir)
    copy_files(os.path.join(this_script_dir, 'train_rt.py'), output_dir)
    copy_files(config_path, output_dir, anno='-copy')

    pretrain_param = configs.get('PretrainParam')
    if pretrain_param is not None and pretrain_param != '':
        load_param_from_file(model,
                             pretrain_param,
                             partially=True,
                             module_namelist=None,
                             logger_name='RT')

    model = model.to(device)
    model.train()

    optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad),
                                 LR,
                                 weight_decay=configs['TRAINING_HYPER_PARAM']['weight_decay'])
    scheduler = make_lr_scheduler(optimizer=optimizer, steps=configs['TRAINING_HYPER_PARAM']['LR_STEPS'],
                                  warmup_iters=configs['TRAINING_HYPER_PARAM']['warmup_iters'], configs=configs)

    if pretrain_param is not None and configs['TRAINING_HYPER_PARAM']['resume']:
        checkpoint = torch.load(pretrain_param, map_location=torch.device("cpu"),
                                pickle_module=dill)
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])

    meters = MetricLogger(delimiter="  ", )
    max_iter = EPOCH * len(train_dataloader)

    start_iter = 0
    start_training_time = time.time()
    end = time.time()
    iteration_best = -1
    best_test_res = 9999999999
    best_model = None

    for epoch in range(EPOCH):
        if configs['UsedModelCFG']['model_name'] == "LSTMTransformer":
            # transform to LSTM + transform end to end finetune mode.
            if epoch >= configs['TRAINING_HYPER_PARAM']['transformer_on_epoch']:
                if hasattr(model, "transformer_flag"):
                    if not model.transformer_flag:
                        model.set_transformer()
                        logger.info("set transformer on")
                else:
                    if not model.module.transformer_flag:
                        model.module.set_transformer()
                        logger.info("set transformer on")

        for idx, (inputs, y) in enumerate(train_dataloader):
            iteration = epoch * len(train_dataloader) + idx

            if isinstance(inputs, tuple):
                seq_x, x_hydro, x_rc = inputs

                seq_x = seq_x.to(device)
                x_hydro = x_hydro.to(device)
                x_rc = x_rc.to(device)
                pred_y = model(x1=seq_x, x2=x_hydro, x3=x_rc).squeeze()
            else:
                inputs = inputs.to(device)
                pred_y = model(x1=inputs).squeeze()
            y = y.to(device)
            loss = loss_func(pred_y, y)

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
                # tf_writer_train.write_data(iter='model', meter=[model, (seq_x, x_hydro, x_rc)])
            if iteration % configs['TRAINING_HYPER_PARAM']['save_param_interval'] == 0 and iteration != 0 \
                    or idx == len(train_dataloader) - 2:
                # evaluation during training

                best_test_res, iteration_best, best_model = evaluation(model, device, logger, tf_writer_train,
                                                                       tf_writer_test,
                                                                       loss_func_eval, test_dataloader,
                                                                       train_val_dataloader,
                                                                       iteration, best_test_res,
                                                                       iteration_best, best_model,
                                                                       holdout_dataloader=None,
                                                                       tf_writer_holdout=None,
                                                                       use_holdout=False)

                if use_holdout:
                    evaluation(model, device, logger, tf_writer_train,
                               tf_writer_test,
                               loss_func_eval, test_dataloader,
                               train_val_dataloader,
                               iteration, best_test_res,
                               iteration_best, best_model,
                               holdout_dataloader=holdout_dataloader,
                               tf_writer_holdout=tf_writer_holdout,
                               use_holdout=True)

                save_checkpoint(model, optimizer, scheduler, output_dir, iteration)
                model.train()
                model = model.to(device)
                if use_cuda:
                    torch.cuda.empty_cache()

    tf_writer_test.write_data(iteration_best, best_test_res, 'eval_metric/Best_delta_t95')
    logger.info("best_test_res: %s in iter %s" % (best_test_res, iteration_best))
    save_checkpoint(model, optimizer, scheduler, output_dir, "last_epochless")
    save_checkpoint(best_model, optimizer, scheduler, output_dir, "best_model")

    if use_holdout:
        model = best_model
        model.to(device)
        evaluation(model, device, logger, tf_writer_train, tf_writer_test,
                   loss_func_eval, test_dataloader, train_val_dataloader,
                   iteration=0, best_test_res=None, holdout_dataloader=holdout_dataloader,
                   tf_writer_holdout=tf_writer_holdout,
                   iteration_best=None, best_model=None, use_holdout=True)


def evaluation(model, device, logger, tf_writer_train, tf_writer_test,
               loss_func_eval, test_dataloader, train_val_dataloader,
               iteration, best_test_res, iteration_best, best_model, holdout_dataloader, tf_writer_holdout,
               use_holdout=False):
    model.eval()

    with torch.no_grad():
        if not use_holdout:
            logger.info("start evaluation on iteration: %d" % iteration)
            logger.info(termcolor.colored("performance on training set:", "yellow"))
            training_loss, pearson, \
            delta_t95, hidden_norm = eval(model, loss_func_eval, train_val_dataloader,
                                          logger, device=device, iteration=iteration)

            tf_writer_train.write_data(iteration, pearson, 'eval_metric/pearson')
            tf_writer_train.write_data(iteration, delta_t95, 'eval_metric/delta_t95')
            tf_writer_train.write_data(iteration, training_loss, "loss")

            logger.info(termcolor.colored("performance on validation set:", "yellow"))
            test_loss, pearson, \
            delta_t95, hidden_norm = eval(model, loss_func_eval, test_dataloader, logger, device=device, iteration=iteration)
            if delta_t95 < best_test_res:
                best_test_res = delta_t95
                iteration_best = iteration
                best_model = copy.deepcopy(model)
            else:
                best_test_res = best_test_res
                iteration_best = iteration_best
                best_model = best_model

            tf_writer_test.write_data(iteration, test_loss, "loss")
            tf_writer_test.write_data(iteration, pearson, 'eval_metric/pearson')
            tf_writer_test.write_data(iteration, delta_t95, 'eval_metric/delta_t95')
            return best_test_res, iteration_best, best_model
        else:
            iteration = 0
            logger.info(termcolor.colored("performance on holdout set:", "yellow"))
            holdout_loss, pearson, \
            delta_t95, hidden_norm = eval(model, loss_func_eval, holdout_dataloader, logger, device=device, iteration=iteration)
            tf_writer_holdout.write_data(iteration, pearson, 'eval_metric/pearson')
            tf_writer_holdout.write_data(iteration, delta_t95, 'eval_metric/delta_t95')
            tf_writer_holdout.write_data(iteration, holdout_loss, "loss")


if __name__ == '__main__':
    logger = logging.getLogger("RT")
    try:
        main()
    except Exception as e:
        logger.error("Error", exc_info=True)
