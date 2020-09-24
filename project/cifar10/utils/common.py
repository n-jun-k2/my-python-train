
import torch
import numpy as np
import torchvision


def to_numpy(x: torch.Tensor):
    '''
        torch.Tnesor to ndarray
    '''
    return x.to('cpu').detach().numpy().copy()


def to_tensor(x: np.ndarray)
    x = x if x.dtype is np.float32 else x.astype(np.float32)
    return torch.from_numpy(x).clone()


def CIFAR10_slice(x: torchvision.datasets.cifar.CIFAR10, start: int, end: int):
    '''
    return:
        image_list: tuple,
        class_list: tuple 
    '''
    retuzip(*[x[i] for i in range(start, end)])