"""
Functions for distrubtued training in pytorch

Created on Thu Sep 28 16:12:44 2023
"""

import torch.distributed as dist
import os
import torch as T


def setup(rank: int, world_size: int) -> None:
    """
    Initalises nccl backend for distributed training communication/
    nccl allows for fast communication of GPU tensors

    args:
        rank (int): rank of process calling function
        world_size (int): Number of processes used for distributed training

    returns:
        None
    """
    if world_size > 0:
        os.environ["MASTER_ADDR"] = (
            "127.0.0.1"  # Set to the IP address of the master node
        )
        os.environ["MASTER_PORT"] = "29500"  # Set to an unused port

        dist.init_process_group(
            backend="nccl", init_method="env://", rank=rank, world_size=world_size
        )


def cleanup() -> None:
    dist.destroy_process_group()  # closes distributed backend


def is_dist_avail_and_initalised() -> bool:
    return (
        True if dist.is_available() and dist.is_initialized() else False
    )  # checks if code is currently being ran distributed


def get_rank() -> int:
    return (
        0 if not is_dist_avail_and_initalised() else dist.get_rank()
    )  # get rank of current process


def is_main_process() -> bool:
    return get_rank() == 0  # main process is typically assinged rank 0


def gather_mean(metric, world_size):
    if world_size is None or world_size == 0:
        return metric
    else:
        dist.all_reduce(metric, op=dist.ReduceOp.SUM)
        return metric / world_size


def gather_concat(tensor, world_size):
    if world_size is None or world_size == 0:
        return tensor
    else:
        empty_tensor = [T.empty_like(tensor) for _ in range(world_size)]
        dist.all_gather(empty_tensor, tensor)
        filled = T.cat(empty_tensor, dim=0)
        return filled
