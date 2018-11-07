# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import logging
import os

import torch


class Checkpoint(object):
    def __init__(self, model, optimizer, scheduler, save_dir, local_rank):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.save_dir = save_dir
        self.local_rank = local_rank

    def __call__(self, name, **kwargs):
        if not self.save_dir:
            return

        if self.local_rank > 0:
            return

        data = {}
        data["model"] = self.model.state_dict()
        data["optimizer"] = self.optimizer.state_dict()
        data["scheduler"] = self.scheduler.state_dict()
        data.update(kwargs)

        save_file = os.path.join(self.save_dir, "{}.pth".format(name))

        logger = logging.getLogger(__name__)
        logger.info("Saving checkpoint to {}".format(save_file))
        torch.save(data, save_file)

    def load(self, f):
        logger = logging.getLogger(__name__)
        logger.info("Loading checkpoint from {}".format(f))
        checkpoint = torch.load(f)
        self.model.load_state_dict(checkpoint.pop("model"))
        self.optimizer.load_state_dict(checkpoint.pop("optimizer"))
        self.scheduler.load_state_dict(checkpoint.pop("scheduler"))

        # extra arguments that were stored
        return checkpoint
