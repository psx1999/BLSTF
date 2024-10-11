from typing import Dict, Union
from packaging import version

import easytorch


def launch_training(cfg: Union[Dict, str], gpus: str = None, node_rank: int = 0):
    easytorch_version = easytorch.__version__
    if version.parse(easytorch_version) >= version.parse("1.3"):
        easytorch.launch_training(cfg=cfg, devices=gpus, node_rank=node_rank)
    else:
        easytorch.launch_training(cfg=cfg, gpus=gpus, node_rank=node_rank)
