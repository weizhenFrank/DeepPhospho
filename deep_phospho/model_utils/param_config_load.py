import logging
import os
from collections import OrderedDict
import ipdb
import dill
import torch
from glob import glob
import copy

import sys


def load_config(config_path):
    config_dir = os.path.dirname(config_path)
    config_file = os.path.basename(config_path)
    config_file_name = os.path.splitext(os.path.basename(config_path))[0]
    sys.path.insert(-1, config_dir)
    cfg = {}
    try:
        exec(f'import {config_file_name} as cfg', {}, cfg)
    except ModuleNotFoundError:
        raise FileNotFoundError(f'Not find the input config file {config_file} with basename {config_file_name} in {config_dir}')
    return cfg['cfg']


def load_param_from_file(model, f: str, partially=False, module_namelist=None, logger_name='IonIntensity'):
    logger = logging.getLogger(logger_name)
    if partially:
        logger.info("partially load weight from %s" % f)
        model = load_weight_partially_from_file(model, f, module_namelist, logger_name)
    else:
        logger.info("load weight from %s" % f)
        state_dict = torch.load(f, map_location=torch.device("cpu"), pickle_module=dill)
        state_dict = state_dict['model']
        model = load_state_dict(model, state_dict)

    return model


# def save_model(model, output_dir, epoch):
#     param_dir = os.path.join(output_dir, "ckpts")
#     if not os.path.exists(param_dir):
#         os.makedirs(param_dir)
#
#     data = {}
#     data["model"] = model.state_dict()
#
#     save_file = os.path.join(param_dir, "{}.pth".format(epoch))
#     print("Saving checkpoint to {}".format(save_file))
#     torch.save(data, save_file, pickle_module=dill)

def save_checkpoint(model, optimizer, scheduler, output_dir, epoch):
    param_dir = os.path.join(output_dir, "ckpts")
    if not os.path.exists(param_dir):
        os.makedirs(param_dir)

    data = {}
    data["model"] = model.state_dict()
    data['optimizer'] = optimizer.state_dict()
    data['scheduler'] = scheduler.state_dict()

    save_file = os.path.join(param_dir, "{}.pth".format(epoch))
    print("Saving checkpoint to {}".format(save_file))
    torch.save(data, save_file, pickle_module=dill)

#
# def load_weight_partially_from_file(model, f: str, module_namelist):
#     logger = logging.getLogger("DeepRT.load_param")
#     state_dict = torch.load(f, map_location=torch.device("cpu"), pickle_module=dill)['model']
#     own_state = model.state_dict()
#     for name, param in state_dict.items():
#         if name.startswith('module'):
#             name = name.strip('module.')
#         try:
#             skip = True
#             if module_namelist is not None:
#                 for to_load_name in module_namelist:
#                     if to_load_name.startswith('module'):
#                         to_load_name = to_load_name.strip('module.')
#                     if to_load_name in name:
#                         skip = False
#
#             else:
#                 skip = False
#             if skip:
#                 continue
#
#             if name not in own_state:
#                 logger.info('[Missed]: {}'.format(name))
#                 continue
#             if isinstance(param, torch.nn.Parameter):
#                 # backwards compatibility for serialized parameters
#                 param = param.data
#             own_state[name].copy_(param)
#             print("[Copied]: {}".format(name))
#             logger.info("[Copied]: {}".format(name))
#         except RuntimeError:
#             logger.info('[Missed] Size Mismatch... : {}'.format(name))
#             print('[Missed] Size Mismatch... : {}'.format(name))
#     logger.info("load the pretrain model %s" % f)
#
#     return model


def load_weight_partially_from_file(model, f: str, module_namelist, logger_name='IonIntensity'):
    logger = logging.getLogger(logger_name)
    state_dict = torch.load(f, map_location=torch.device("cpu"), pickle_module=dill)['model']
    # ipdb.set_trace()
    own_state = model.state_dict()
    if module_namelist is not None:

        for to_load_name in module_namelist:
            param = state_dict[to_load_name]
            if to_load_name.startswith('module'):
                # ipdb.set_trace()
                to_load_name = to_load_name.replace('module.', '')
                # ipdb.set_trace()
            try:
                if isinstance(param, torch.nn.Parameter):
                    param = param.data
                    # backwards compatibility for serialized parameters
                own_state[to_load_name].copy_(param)
                # print("[Copied]: {}".format(to_load_name))
                logger.info("[Copied]: {}".format(to_load_name))

            except RuntimeError:
                logger.info('[Missed] Size Mismatch... : {}'.format(to_load_name))

    else:

        for name, param in state_dict.items():
            if name.startswith('module'):
                name = name.replace('module.', '')
            try:

                if name not in own_state:
                    logger.info('[Missed]: {}'.format(name))
                    continue
                if isinstance(param, torch.nn.Parameter):
                    # backwards compatibility for serialized parameters
                    param = param.data
                own_state[name].copy_(param)
                # ipdb.set_trace()
                # print("[Copied]: {}".format(name))
                logger.info("[Copied]: {}".format(name))
            except RuntimeError:
                logger.info('[Missed] Size Mismatch... : {}'.format(name))
        logger.info("load the pretrain model")

    return model


def align_and_update_state_dicts(model_state_dict, loaded_state_dict):
    """
    Strategy: suppose that the models that we will create will have prefixes appended
    to each of its keys, for example due to an extra level of nesting that the original
    pre-trained weights from ImageNet won't contain. For example, model.state_dict()
    might return backbone[0].body.res2.conv1.weight, while the pre-trained model contains
    res2.conv1.weight. We thus want to match both parameters together.
    For that, we look for each model weight, look among all loaded keys if there is one
    that is a suffix of the current weight name, and use it if that's the case.
    If multiple matches exist, take the one with longest size
    of the corresponding name. For example, for the same model as before, the pretrained
    weight file can contain both res2.conv1.weight, as well as conv1.weight. In this case,
    we want to match backbone[0].body.conv1.weight to conv1.weight, and
    backbone[0].body.res2.conv1.weight to res2.conv1.weight.
    """
    current_keys = sorted(list(model_state_dict.keys()))
    loaded_keys = sorted(list(loaded_state_dict.keys()))
    # get a matrix of string matches, where each (i, j) entry correspond to the size of the
    # loaded_key string, if it matches
    match_matrix = [
        len(j) if i.endswith(j) else 0 for i in current_keys for j in loaded_keys
    ]
    match_matrix = torch.as_tensor(match_matrix).view(
        len(current_keys), len(loaded_keys)
    )
    max_match_size, idxs = match_matrix.max(1)
    # remove indices that correspond to no-match
    idxs[max_match_size == 0] = -1

    # used for logging
    max_size = max([len(key) for key in current_keys]) if current_keys else 1
    max_size_loaded = max([len(key) for key in loaded_keys]) if loaded_keys else 1
    log_str_template = "{: <{}} loaded from {: <{}} of shape {}"
    logger = logging.getLogger("IonIntensity")
    for idx_new, idx_old in enumerate(idxs.tolist()):
        if idx_old == -1:
            continue
        key = current_keys[idx_new]
        key_old = loaded_keys[idx_old]
        model_state_dict[key] = loaded_state_dict[key_old]
        logger.info(
            log_str_template.format(
                key,
                max_size,
                key_old,
                max_size_loaded,
                tuple(loaded_state_dict[key_old].shape),
            )
        )


def strip_prefix_if_present(state_dict, prefix):
    keys = sorted(state_dict.keys())
    if not all(key.startswith(prefix) for key in keys):
        return state_dict
    stripped_state_dict = OrderedDict()
    for key, value in state_dict.items():
        stripped_state_dict[key.replace(prefix, "")] = value
    return stripped_state_dict


def load_state_dict(model, loaded_state_dict):
    # ipdb.set_trace()
    model_state_dict = model.state_dict()

    # if the state_dict comes from a model that was wrapped in a
    # DataParallel or DistributedDataParallel during serialization,
    # remove the "module" prefix before performing the matching
    loaded_state_dict = strip_prefix_if_present(loaded_state_dict, prefix="module.")
    align_and_update_state_dicts(model_state_dict, loaded_state_dict)
    # use strict loading
    model.load_state_dict(model_state_dict)
    return model


def load_average_model(model, saved_models_path, iterations: int,):

    model_files = [os.path.join(saved_models_path, iteration_model) for iteration_model in os.listdir(saved_models_path)]
    # ipdb.set_trace()
    model_files.sort(key=os.path.getmtime)
    to_load_models = model_files[-iterations:]
    models_weight = [torch.load(weight_path, map_location=torch.device("cpu"), pickle_module=dill)['model'] for weight_path in to_load_models]

    own_state = copy.deepcopy(model.state_dict())
    for name in own_state:
        # ipdb.set_trace()
        own_state[name] = (1/iterations) * (models_weight[0][name])
        if iterations >= 2:
            for idx in range(1, iterations):
                own_state[name] += (1/iterations)*models_weight[idx][name]

    model.load_state_dict(own_state)

    return model

