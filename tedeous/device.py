"""Module for working with device mode"""

from typing import Any
import torch


def solver_device(device: str):
    """
    Sets the default device (CPU or CUDA) for subsequent PyTorch tensor operations.
    
    This ensures that all tensors created after calling this function will reside on the specified device, 
    streamlining the process of utilizing available hardware resources for solving differential equations.
    
    Args:
        device (str):  Desired device mode ('cuda', 'gpu', or 'cpu').  If 'cuda' or 'gpu' are specified and CUDA is available, 
                       the device is set to CUDA; otherwise, it defaults to CPU.
    
    Returns:
        None: This function does not return any value, but sets the default device for PyTorch.
    """
    if device in ['cuda', 'gpu'] and torch.cuda.is_available():
        print('CUDA is available and used.')
        return torch.set_default_device('cuda')
    elif device in ['cuda', 'gpu'] and not torch.cuda.is_available():
        print('CUDA is not available, cpu is used!')
        return torch.set_default_device('cpu')
    else:
        print('Default cpu processor is used.')
        return torch.set_default_device('cpu')


def check_device(data: Any):
    """
    Ensures that the input data (either a tensor or something convertible to a tensor) resides on the expected device.
    
        This is crucial for maintaining consistency in computations within the neural network solver,
        preventing device-related errors during training and evaluation. By ensuring all data is on the
        same device, we guarantee compatibility during tensor operations.
    
    Args:
        data (Any): The input data, which can be a PyTorch tensor or a data structure that can be converted into a tensor.
    
    Returns:
        Any: The input data, converted to a PyTorch tensor and moved to the correct device if necessary.
             If the input is already a tensor on the correct device, it is returned unchanged.
    
    Raises:
        TypeError: If the input data cannot be converted to a PyTorch tensor.
    """
    device = torch.tensor([0.]).device
    if isinstance(data, torch.Tensor):
        if data.device != device:
            return data.to(device)
        return data
    else:
        try:
            tensor_data = torch.tensor(data)
            return tensor_data.to(device)
        except Exception as e:
            raise TypeError(f"Cannot convert data to tensor. Ensure it's a compatible type. Error: {e}")


def device_type():
    """
    Return the default device type used for computations.
    
    This is important for ensuring that the neural network models and data
    are on the same device, enabling efficient training and inference
    when approximating solutions to differential equations.
    
    Args:
        None
    
    Returns:
        str: The type of the default device (e.g., 'cpu' or 'cuda').
    """
    return torch.tensor([0.]).device.type



