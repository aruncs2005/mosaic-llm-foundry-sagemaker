# Copyright 2022 MosaicML Diffusion authors
# SPDX-License-Identifier: Apache-2.0

"""Run training."""

from omegaconf import OmegaConf as om
from train import main
import torch.distributed as dist
import os

import socket
import json
import argparse

num_gpus = int(os.environ['SM_NUM_GPUS'])
current_node = os.environ['SM_CURRENT_HOST']
    
hosts = json.loads(os.environ['SM_HOSTS'])
leader = socket.gethostbyname(hosts[0])
num_nodes = len(hosts)

#set environment variables for mosaic trainer
os.environ['LOCAL_WORLD_SIZE'] = str(num_gpus)
os.environ["NODE_RANK"] = str(hosts.index(current_node))
os.environ['HYDRA_FULL_ERROR'] = '1'


# #set environment variables for mosaic composer trainer
local_rank = int(os.environ["LOCAL_RANK"])
print(f"local rank {local_rank}")

parser = argparse.ArgumentParser()
parser.add_argument('--config_path', type=str, default='yamls')
parser.add_argument('--config_name', type=str, default='mpt-7b.yaml')
parser.add_argument('--backend',type=str,default="nccl")
args = parser.parse_args()

print(os.environ)



def run() -> None:
    """Hydra wrapper for train."""
    yaml_path = os.path.join(args.config_path , args.config_name)
    with open(yaml_path) as f:
        yaml_cfg = om.load(f)
    return main(yaml_cfg)

if __name__ == '__main__':
    if args.backend == "smddp":
        import patch_init
    dist.init_process_group(backend=args.backend)
    run()