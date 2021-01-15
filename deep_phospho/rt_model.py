import os
import time
import datetime
import copy
import logging
import random

import dill
import ipdb
import numpy as np
import termcolor

import torch
from torch.utils.data import DataLoader

from deep_phospho.configs import config_main as cfg

from deep_phospho.model_dataset.preprocess_input_data import RTdata, Dictionary
from deep_phospho.model_dataset.dataset import IonDataset, collate_fn

from deep_phospho.models.EnsembelModel import LSTMTransformer

from deep_phospho.model_utils.rt_eval import eval
from deep_phospho.model_utils.logger import MetricLogger, setup_logger, save_config, TFBoardWriter
from deep_phospho.model_utils.lr_scheduler import make_lr_scheduler
from deep_phospho.model_utils.utils_functions import copy_files, get_loss, show_params_status, get_parser
from deep_phospho.model_utils.param_config_load import save_checkpoint, load_param_from_file

logging.basicConfig(level=logging.INFO)
SEED = 666
torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.backends.cudnn.deterministic = True  # To test dilated conv, the result shows set true make conv dilation slower 7X
# change as True to test the Jeff result huge difference
torch.autograd.set_detect_anomaly(True)


class RTModel(object):
    def __init__(self, config):
        self.args = self.get_args()

    def get_args(self):
        return get_parser('RT prediction')


def main():
    if cfg.TRAINING_HYPER_PARAM['resume']:
        resume = 'RESUME'
    else:
        resume = ''
    comment = "%s-%s-%s%s" % (
        cfg.data_name,
        cfg.MODEL_CFG['model_name'],
        args.exp_name, resume)

    now = datetime.datetime.now()
    time_str = now.strftime("%Y-%m-%d_%H-%M-%S")
    instance_name = f'{time_str}_{comment}'
    output_dir = os.path.join('../result/RT', instance_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if args.pretrain_param is not None:
        cfg.TRAINING_HYPER_PARAM['pretrain_param'] = args.pretrain_param
    if args.ad_hoc is not None:
        """
        Here is ad_hoc
        """
        cfg.MODEL_CFG['num_encd_layer'] = int(args.ad_hoc)
    if args.GPU is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.GPU
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = cfg.TRAINING_HYPER_PARAM['GPU_INDEX']
    tf_writer_train = TFBoardWriter(output_dir, type='train')
    tf_writer_test = TFBoardWriter(output_dir, type="val")
    logger = setup_logger("RT", output_dir)
    dictionary = Dictionary()

    print("Preparing dataset")
    RTtrain = RTdata(cfg.TRAIN_DATA_CFG,
                     dictionary=dictionary)
    RTtest = RTdata(cfg.TEST_DATA_CFG,
                    dictionary=dictionary)
    if args.use_holdout:
        RTholdout = RTdata(cfg.HOLDOUT_DATA_CFG,
                           dictionary=dictionary)

        holdout_dataset = IonDataset(RTholdout)
        holdout_dataloader = DataLoader(
            dataset=holdout_dataset,
            batch_size=256,
            shuffle=False,
            num_workers=2,
            collate_fn=collate_fn
        )
        tf_writer_holdout = TFBoardWriter(output_dir, type='test')

    train_dataset = IonDataset(RTtrain)
    test_dataset = IonDataset(RTtest)

    train_dataloader = DataLoader(dataset=train_dataset,
                                  batch_size=cfg.TRAINING_HYPER_PARAM['BATCH_SIZE'],
                                  shuffle=False,
                                  num_workers=4,
                                  collate_fn=collate_fn, drop_last=True)

    train_val_dataloader = DataLoader(dataset=train_dataset,
                                      batch_size=1024,
                                      shuffle=False,
                                      num_workers=2,
                                      collate_fn=collate_fn)

    test_dataloader = DataLoader(dataset=test_dataset,
                                 shuffle=False,
                                 batch_size=1024,
                                 num_workers=2,
                                 collate_fn=collate_fn)

    loss_func = get_loss()

    loss_func_eval = copy.deepcopy(loss_func)
    logger.info(save_config(cfg, save_dir=output_dir))
    EPOCH = cfg.TRAINING_HYPER_PARAM['EPOCH']
    LR = cfg.TRAINING_HYPER_PARAM['LR']

    if cfg.MODEL_CFG['model_name'] == "LSTMTransformer":
        cfg_to_load = copy.deepcopy(cfg.MODEL_CFG)

        model = LSTMTransformer(
            # ntoken=Iontrain.N_aa,
            RT_mode=cfg.Mode == "RT" or cfg.Mode == "Detect",
            ntoken=RTtrain.N_aa,
            # for prosit, it has 0-21
            **cfg_to_load,
        )

    else:
        raise RuntimeError("No valid model name given.")

    logger.info(str(model))
    logger.info("model parameters statuts: \n%s" % show_params_status(model))
    copy_files("deep_phospho/models/ion_model.py", output_dir)
    copy_files("deep_phospho/models/EnsembelModel.py", output_dir)
    copy_files("main.py", output_dir)
    copy_files("deep_phospho/configs", output_dir)

    if cfg.TRAINING_HYPER_PARAM.get("pretrain_param") is not None:
        if cfg.TRAINING_HYPER_PARAM.get("pretrain_param") != '':
            load_param_from_file(model,
                                 cfg.TRAINING_HYPER_PARAM['pretrain_param'],
                                 partially=True,
                                 module_namelist=None, logger_name='RT')

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

    if cfg.TRAINING_HYPER_PARAM.get("pretrain_param") is not None and cfg.TRAINING_HYPER_PARAM['resume']:
        checkpoint = torch.load(cfg.TRAINING_HYPER_PARAM['pretrain_param'], map_location=torch.device("cpu"),
                                pickle_module=dill)
        ipdb.set_trace()
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
        if cfg.MODEL_CFG['model_name'] == "LSTMTransformer":
            # transform to LSTM + transform end to end finetune mode.
            if epoch >= cfg.TRAINING_HYPER_PARAM['transformer_on_epoch']:
                # ipdb.set_trace()
                if hasattr(model, "transformer_flag"):
                    if not model.transformer_flag:
                        # ipdb.set_trace()
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

                seq_x = seq_x.cuda()
                x_hydro = x_hydro.cuda()
                x_rc = x_rc.cuda()
                pred_y = model(x1=seq_x, x2=x_hydro, x3=x_rc).squeeze()
            else:
                # ipdb.set_trace()
                inputs = inputs.cuda()
                pred_y = model(x1=inputs).squeeze()
            y = y.cuda()
            # ipdb.set_trace()
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
                # tf_writer_train.write_data(iter='model', meter=[model, (seq_x, x_hydro, x_rc)])
            if iteration % cfg.TRAINING_HYPER_PARAM['save_param_interval'] == 0 and iteration != 0 \
                    or idx == len(train_dataloader) - 2:
                # evaluation during training
                if args.use_holdout:
                    best_test_res, iteration_best, best_model = evaluation(model, logger, tf_writer_train,
                                                                           tf_writer_test,
                                                                           loss_func_eval, test_dataloader,
                                                                           train_val_dataloader,
                                                                           iteration, best_test_res,
                                                                           iteration_best, best_model,
                                                                           holdout_dataloader, tf_writer_holdout,
                                                                           use_holdout=True)
                else:
                    best_test_res, iteration_best, best_model = evaluation(model, logger, tf_writer_train,
                                                                           tf_writer_test,
                                                                           loss_func_eval, test_dataloader,
                                                                           train_val_dataloader,
                                                                           iteration, best_test_res,
                                                                           iteration_best, best_model,
                                                                           holdout_dataloader=None,
                                                                           tf_writer_holdout=None,
                                                                           use_holdout=False)

                save_checkpoint(model, optimizer, scheduler, output_dir, iteration)
                model.train()
                model = model.cuda()
                torch.cuda.empty_cache()

    tf_writer_test.write_data(iteration_best, best_test_res, 'eval_metric/Best_delta_t95')
    logger.info("best_test_res: %s in iter %s" % (best_test_res, iteration_best))
    save_checkpoint(model, optimizer, scheduler, output_dir, "last_epochless")
    save_checkpoint(best_model, optimizer, scheduler, output_dir, "best_model")

    if args.use_holdout:
        model = best_model
        model.cuda()
        evaluation(model, logger, tf_writer_train, tf_writer_test,
                   loss_func_eval, test_dataloader, train_val_dataloader,
                   iteration=0, best_test_res=None, holdout_dataloader=holdout_dataloader,
                   tf_writer_holdout=tf_writer_holdout,
                   iteration_best=None, best_model=None, use_holdout=True)


def evaluation(model, logger, tf_writer_train, tf_writer_test,
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
                                          logger, iteration)

            tf_writer_train.write_data(iteration, pearson, 'eval_metric/pearson')
            tf_writer_train.write_data(iteration, delta_t95, 'eval_metric/delta_t95')
            tf_writer_train.write_data(iteration, training_loss, "loss")

            logger.info(termcolor.colored("performance on validation set:", "yellow"))
            test_loss, pearson, \
            delta_t95, hidden_norm = eval(model, loss_func_eval, test_dataloader, logger, iteration)
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
            delta_t95, hidden_norm = eval(model, loss_func_eval, holdout_dataloader,
                                          logger)
            tf_writer_holdout.write_data(iteration, pearson, 'eval_metric/pearson')
            tf_writer_holdout.write_data(iteration, delta_t95, 'eval_metric/delta_t95')
            tf_writer_holdout.write_data(iteration, holdout_loss, "loss")


if __name__ == '__main__':
    logger = logging.getLogger("RT")
    try:
        main()
    except Exception as e:
        logger.error("Error", exc_info=True)
