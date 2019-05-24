# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import time
from collections import defaultdict
from collections import deque
from datetime import datetime

import torch

from .comm import is_main_process


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20):
        self.deque = deque(maxlen=window_size)
        self.series = []
        self.total = 0.0
        self.count = 0

    def update(self, value):
        self.deque.append(value)
        self.series.append(value)
        self.count += 1
        self.total += value

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
        return self.total / self.count


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

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {:.4f} ({:.4f})".format(name, meter.median, meter.global_avg)
            )
        return self.delimiter.join(loss_str)


class TensorboardLogger(MetricLogger):
    def __init__(self,
                 log_dir,
                 start_iter=0,
                 delimiter='\t'):

        super(TensorboardLogger, self).__init__(delimiter)
        self.iteration = start_iter
        self.writer = self._get_tensorboard_writer(log_dir)

    @staticmethod
    def _get_tensorboard_writer(log_dir):
        try:
            from tensorboardX import SummaryWriter
        except ImportError:
            raise ImportError(
                'To use tensorboard please install tensorboardX '
                '[ pip install tensorflow tensorboardX ].'
            )

        if is_main_process():
            timestamp = datetime.fromtimestamp(time.time()).strftime('%Y%m%d-%H:%M')
            tb_logger = SummaryWriter('{}-{}'.format(log_dir, timestamp))
            return tb_logger
        else:
            return None

    def update(self, ** kwargs):
        super(TensorboardLogger, self).update(**kwargs)
        if self.writer:
            for k, v in kwargs.items():
                if isinstance(v, torch.Tensor):
                    v = v.item()
                assert isinstance(v, (float, int))
                self.writer.add_scalar(k, v, self.iteration)
            self.iteration += 1
