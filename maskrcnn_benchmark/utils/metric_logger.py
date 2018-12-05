# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from collections import defaultdict
from collections import deque

import torch
import time, os, io
from datetime import datetime
from .comm import is_main_process
import matplotlib.pyplot as plt
from matplotlib import patches

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
        return object.__getattr__(self, attr)

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {:.4f} ({:.4f})".format(name, meter.median, meter.global_avg)
            )
        return self.delimiter.join(loss_str)
        

class TensorboardLogger(MetricLogger):
    def __init__(self, log_dir='logs', start_iter=0, delimiter='\t'):
        
        super(TensorboardLogger, self).__init__(delimiter)
        self.iteration = start_iter
        self.writer = self._get_tensorboard_writer(log_dir)
        
    @staticmethod
    def _get_tensorboard_writer(log_dir):
        try:
            from tensorboardX import SummaryWriter
        except:
            raise ImportError('TensorboardX not installed!')
            
        if is_main_process():
            timestamp = datetime.fromtimestamp(time.time()).strftime('%Y%m%d-%H:%M')
            tb_logger = SummaryWriter(os.path.join(log_dir, timestamp))
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
            
    def update_image(self, image, preds):
        image = image.cpu().numpy()
        boxes = preds.bbox.cpu().numpy()
        cats = preds.get_field('labels')
        
        if self.writer:
            for cat in cats.unique():
                if cat == 0:
                    continue
                fig, ax = plt.figure(), plt.gca()
                ax.imshow(image.transpose(1, 2, 0))
                
                for i in range(len(cats)):
                    if cats[i] == cat:
                        x1, y1, x2, y2 = boxes[i]
                        ax.add_patch(
                            patches.Rectangle(
                                (x1, y1),
                                x2 - x1,
                                y2 - y1,
                                edgecolor='r',
                                linewidth=1,
                                fill=False
                            )
                        )
                        
                plt.axis('scaled')
                plt.tight_layout()
                        
                self.writer.add_figure('train/image/{}'.format(cat), fig, self.iteration)

    