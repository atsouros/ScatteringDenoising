"""
File containing basic functions
"""
import numpy as np
import torch
from torch.fft import fft2, ifft2
import psutil

def to_numpy(data):
    """Converts a tensor/array/list to numpy array. Recurse over dictionaries and tuples. Values are left as-is."""
    if torch.is_tensor(data):
        return data.detach().cpu().numpy()
    elif isinstance(data, np.ndarray):
        return data
    elif isinstance(data, list):
        return np.array(data)
    elif isinstance(data, dict):
        return {k: to_numpy(v) for k, v in data.items()}
    elif isinstance(data, tuple):
        return tuple(to_numpy(v) for v in data)
    return data

def to_torch(data, device=None, precision="single"):
    """
    Converts input data to pytorch tensor. The tensor is automatically sent to the specified device,
    and using the specified precision.
    
    Takes as an input a numpy array or a pytorch tensor.

    Parameters
    ----------
    data : array or tensor or list
        Input data.
    device : str, optional
        Target device. The default is None.
    precision : str, optional
        Either 'single' or 'double' precision. The default is np.float32.

    Returns
    -------
    tensor
        Tensor with the same shape than the input data.

    """
    ret = None
    
    if isinstance(data, list): # List
        return to_torch(np.array(data), device=device, precision=precision)
    elif isinstance(data, np.ndarray): # Numpy array
        ret = torch.from_numpy(data.astype(get_precision_type(precision, module="numpy", cplx=np.iscomplexobj(data))))
    elif isinstance(data, torch.Tensor):
        ret = data
    else:
        raise Exception("Unknown data type!")
    if device is not None:
        ret = ret.to(device)
    return ret.contiguous()


def get_memory_available(device):
    """
    Returns available memory in bytes.

    Parameters
    ----------
    device : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    if device == "cpu" or device == "mps":
        return psutil.virtual_memory().available
    else:
        if isinstance(device, torch.device):
            device = device.index
        t = torch.cuda.get_device_properties(device).total_memory
        a = get_gpu_memory_map()[device]
        return t - (a - (torch.cuda.memory_reserved(device) - torch.cuda.memory_allocated(device))  // (1024 ** 2)) * 1024 ** 2


def fft(z):
    """
    Torch 2D FFT wrapper. No padding. The FFT is applied to the 2 last dimensions.

    Parameters
    ----------
    z : tensor
        Input.

    Returns
    -------
    tensor
        Output.

    """
    return fft2(z, s=(-1, -1))


def ifft(z):
    """
    Torch 2D IFFT wrapper. No padding. The IFFT is applied to the 2 last dimensions.

    Parameters
    ----------
    z : tensor
        Input.

    Returns
    -------
    tensor
        Output.

    """
    return ifft2(z, s=(-1, -1))
