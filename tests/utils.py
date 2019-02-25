from __future__ import absolute_import, division, print_function, unicode_literals

# Set up custom environment before nearly anything else is imported
# NOTE: this should be the first import (no not reorder)
from maskrcnn_benchmark.utils.env import setup_environment  # noqa F401 isort:skip
import env_tests.env as env_tests

import os
import copy

from maskrcnn_benchmark.config import cfg as g_cfg


def get_config_root_path():
    return env_tests.get_config_root_path()


def load_config(rel_path):
    ''' Load config from file path specified as path relative to config_root '''
    cfg_path = os.path.join(env_tests.get_config_root_path(), rel_path)
    return load_config_from_file(cfg_path)


def load_config_from_file(file_path):
    ''' Load config from file path specified as absolute path '''
    ret = copy.deepcopy(g_cfg)
    ret.merge_from_file(file_path)
    return ret
