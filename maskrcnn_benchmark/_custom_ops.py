import os

import torch

torch.ops.load_library(os.path.join(os.path.dirname(__file__), 'lib', 'libmaskrcnn_benchmark_customops.so'))
