import logging
import os
import sys
import json
from collections import defaultdict, OrderedDict, deque

import torch
from torch.utils.tensorboard import SummaryWriter


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=50):
        self.deque = deque(maxlen=window_size)
        self.series = []
        self.total = 0.0
        self.count = 0

    def update(self, value):
        self.deque.append(value)
        self.series.append(value)
        self.count += 1
        self.total += value

    def reset(self):
        self.series = []
        self.total = 0.0
        self.count = 0

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque))
        return d.mean().item()

    @property
    def global_avg(self):
        return torch.tensor(self.series).mean().item()


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def reset(self):
        for name, meter in self.meters.items():
            meter.reset()

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {:.4f} ({:.4f}, {:.4f})\n".format(name, meter.median,
                                                       meter.avg, meter.global_avg)
            )
        return self.delimiter.join(loss_str)


def save_config(conf, save_dir):
    conf_dict = conf_to_dict(conf)
    with open(os.path.join(save_dir, "config.json"), 'w') as f:
        conf_s = json.dumps(conf_dict, indent=4)
        f.write(conf_s)
    return conf_s


def conf_to_dict(conf):
    conf_dict = OrderedDict()
    d = conf.__dict__
    for each in d:
        if each[0] != '_':
            conf_dict[each] = d[each]
    return conf_dict


TFBoardHandler_LEVEL = 4


def setup_logger(name, save_dir, filename="log.txt"):
    # remove root Handler
    logger = logging.getLogger()
    for each in logger.handlers:
        logger.removeHandler(each)

    logger = logging.getLogger(name)
    logger.setLevel(TFBoardHandler_LEVEL)
    # don't log results for the non-master process
    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s:\n%(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    if save_dir is not None:
        fh = logging.FileHandler(os.path.join(save_dir, filename))
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger


class TFBoardWriter:
    def __init__(self, log_dir, type=None):
        if type is not None:
            tfbd_dir = os.path.join(log_dir, 'tfboard', type)
        else:
            tfbd_dir = os.path.join(log_dir, 'tfboard')

        if not os.path.exists(tfbd_dir):
            os.makedirs(tfbd_dir)

        self.tf_writer = SummaryWriter(log_dir=tfbd_dir,
                                       flush_secs=20)
        self.enable = True

    def write_data(self, iter, meter, key=None):
        if isinstance(iter, str):
            model = meter[0]
            input = meter[1]
            self.tf_writer.add_graph(model, input)
        else:
            if key is None:
                for each in meter.keys():
                    val = meter[each]
                    if isinstance(val, SmoothedValue):
                        val = val.avg
                    self.tf_writer.add_scalar(each, val, iter)
            else:
                self.tf_writer.add_scalar(key, meter, iter)

    def add_pr_curve_raw(self, iter, true_positive_counts, false_positive_counts, true_negative_counts, false_negative_counts, precision, recall, key):
        self.tf_writer.add_pr_curve_raw(key, true_positive_counts, false_positive_counts, true_negative_counts, false_negative_counts, precision, recall, iter)

    def close(self):
        if self.enable:
            self.tf_writer.close()


class TFBoardHandler(logging.Handler):
    def __init__(self, writer):
        logging.Handler.__init__(self, TFBoardHandler_LEVEL)
        self.tf_writer = writer

    def emit(self, record):
        return

    def close(self):
        self.tf_writer.close()
