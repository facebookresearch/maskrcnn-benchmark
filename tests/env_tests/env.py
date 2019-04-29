# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import os


def get_config_root_path():
    ''' Path to configs for unit tests '''
    # cur_file_dir is root/tests/env_tests
    cur_file_dir = os.path.dirname(os.path.abspath(os.path.realpath(__file__)))
    ret = os.path.dirname(os.path.dirname(cur_file_dir))
    ret = os.path.join(ret, "configs")
    return ret
