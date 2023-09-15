
from composer.devices import Device
from composer.utils import dist
from typing import Union
from typing import  Any, List
import torch.distributed as dist1
import torch


def initialize_dist(device: Union[str, Device], timeout: float = 300.0):
    print("ingoring the initialize as SMDDP is already initialized")
    pass


def broadcast_object_list(object_list: List[Any], src: int = 0) -> None:

    if dist.is_available() and dist.is_initialized():
        dist1.broadcast_object_list(object_list, src, device=torch.device('cuda'))
        return
    world_size = dist1.get_world_size()
    if world_size == 1:
        return
    raise RuntimeError(f'The world_size({world_size}) > 1, but the distributed package is not '
                       'available or has not been initialized. Please check you have initialized '
                       'the distributed runtime and that PyTorch has been built with distributed '
                       'support. If calling this function outside Trainer, please ensure that '
                       '`composer.utils.dist.initialize_dist` has been called first.')

init_stub = dist.initialize_dist
broadcast_stub = dist.broadcast_object_list

dist.initialize_dist = initialize_dist
dist.broadcast_object_list = broadcast_object_list