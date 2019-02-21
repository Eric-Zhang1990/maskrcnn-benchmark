import os

import torch

torch.ops.load_library(os.path.join(os.path.dirname(__file__), 'lib', 'custom_ops.cp36-win_amd64.pyd'))
