# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""
This file contains primitives for multi-gpu communication.
This is useful when doing distributed training.
"""

import os
import pickle
import tempfile
import time

import torch


def get_world_size():
    if not torch.distributed.is_initialized():
        return 1
    return torch.distributed.get_world_size()


def get_rank():
    if not torch.distributed.is_initialized():
        return 0
    return torch.distributed.get_rank()


def is_main_process():
    if not torch.distributed.is_initialized():
        return True
    return torch.distributed.get_rank() == 0


def synchronize():
    """
    Helper function to synchronize between multiple processes when
    using distributed training
    """
    if not torch.distributed.is_initialized():
        return
    world_size = torch.distributed.get_world_size()
    rank = torch.distributed.get_rank()
    if world_size == 1:
        return

    def _send_and_wait(r):
        if rank == r:
            tensor = torch.tensor(0, device="cuda")
        else:
            tensor = torch.tensor(1, device="cuda")
        torch.distributed.broadcast(tensor, r)
        while tensor.item() == 1:
            time.sleep(1)

    _send_and_wait(0)
    # now sync on the main process
    _send_and_wait(1)


def _encode(encoded_data, data):
    # gets a byte representation for the data
    encoded_bytes = pickle.dumps(data)
    # convert this byte string into a byte tensor
    storage = torch.ByteStorage.from_buffer(encoded_bytes)
    tensor = torch.ByteTensor(storage).to("cuda")
    # encoding: first byte is the size and then rest is the data
    s = tensor.numel()
    assert s <= 255, "Can't encode data greater than 255 bytes"
    # put the encoded data in encoded_data
    encoded_data[0] = s
    encoded_data[1 : (s + 1)] = tensor


def _decode(encoded_data):
    size = encoded_data[0]
    encoded_tensor = encoded_data[1 : (size + 1)].to("cpu")
    return pickle.loads(bytearray(encoded_tensor.tolist()))


# TODO try to use tensor in shared-memory instead of serializing to disk
# this involves getting the all_gather to work
def scatter_gather(data):
    """
    This function gathers data from multiple processes, and returns them
    in a list, as they were obtained from each process.

    This function is useful for retrieving data from multiple processes,
    when launching the code with torch.distributed.launch

    Note: this function is slow and should not be used in tight loops, i.e.,
    do not use it in the training loop.

    Arguments:
        data: the object to be gathered from multiple processes.
            It must be serializable

    Returns:
        result (list): a list with as many elements as there are processes,
            where each element i in the list corresponds to the data that was
            gathered from the process of rank i.
    """
    # strategy: the main process creates a temporary directory, and communicates
    # the location of the temporary directory to all other processes.
    # each process will then serialize the data to the folder defined by
    # the main process, and then the main process reads all of the serialized
    # files and returns them in a list
    if not torch.distributed.is_initialized():
        return [data]
    synchronize()
    # get rank of the current process
    rank = torch.distributed.get_rank()

    # the data to communicate should be small
    data_to_communicate = torch.empty(256, dtype=torch.uint8, device="cuda")
    if rank == 0:
        # manually creates a temporary directory, that needs to be cleaned
        # afterwards
        tmp_dir = tempfile.mkdtemp()
        _encode(data_to_communicate, tmp_dir)

    synchronize()
    # the main process (rank=0) communicates the data to all processes
    torch.distributed.broadcast(data_to_communicate, 0)

    # get the data that was communicated
    tmp_dir = _decode(data_to_communicate)

    # each process serializes to a different file
    file_template = "file{}.pth"
    tmp_file = os.path.join(tmp_dir, file_template.format(rank))
    torch.save(data, tmp_file)

    # synchronize before loading the data
    synchronize()

    # only the master process returns the data
    if rank == 0:
        data_list = []
        world_size = torch.distributed.get_world_size()
        for r in range(world_size):
            file_path = os.path.join(tmp_dir, file_template.format(r))
            d = torch.load(file_path)
            data_list.append(d)
            # cleanup
            os.remove(file_path)
        # cleanup
        os.rmdir(tmp_dir)
        return data_list
