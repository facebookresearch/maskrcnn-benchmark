# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import unittest
import glob
import os
import utils


class TestConfigs(unittest.TestCase):
    def test_configs_load(self):
        ''' Make sure configs are loadable '''

        cfg_root_path = utils.get_config_root_path()
        files = glob.glob(
            os.path.join(cfg_root_path, "./**/*.yaml"), recursive=True)
        self.assertGreater(len(files), 0)

        for fn in files:
            print('Loading {}...'.format(fn))
            utils.load_config_from_file(fn)


if __name__ == "__main__":
    unittest.main()
