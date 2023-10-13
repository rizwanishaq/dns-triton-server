
import numpy as np
from scipy.signal import resample_poly
from typing import Tuple
import triton_python_backend_utils as pb_utils


def get_input_tensor_as_numpy(request, tensor_name):
    """Get an input tensor as a numpy array.

    Args:
        request (pb_utils.InferenceRequest): The inference request.
        tensor_name (str): The name of the tensor.

    Returns:
        numpy.ndarray: The input tensor as a numpy array.
    """
    return pb_utils.get_input_tensor_by_name(request, tensor_name).as_numpy()[0]


def map_function(s: np.ndarray, fs_orig: int, fs_target: int = 8000) -> np.ndarray:
    """
    Resample audio signal `s` to the target sample rate `fs_target`.

    Args:
        s (ndarray): Input audio signal.
        fs_orig (int): Original sample rate of the input signal.
        fs_target (int, optional): Target sample rate. Default is 8000.

    Returns:
        ndarray: Resampled audio signal.

    Raises:
        ValueError: If the input signal does not have one or two dimensions.

    """
    if s.ndim == 2:
        s = s[:, 0]
    elif s.ndim == 1 and s.dtype == np.int16:
        s = s / 0x8000
    else:
        raise ValueError(
            "Input signal must be 1D or 2D ndarray of type int16.")

    if fs_orig != fs_target:
        s = resample_poly(s, fs_target, fs_orig)

    return s.astype('float32')


def to_int16(data: np.ndarray) -> np.ndarray:
    """
    Convert input array `data` to int16 format.

    Args:
        data (ndarray): Input data.

    Returns:
        ndarray: Converted data in int16 format.

    Raises:
        ValueError: If the input data type is not supported.

    """
    if data.dtype == np.float32:
        return (data * 0x7fff).astype('int16')
    elif data.dtype == np.int16:
        return data
    else:
        raise ValueError("Input data type must be float32 or int16.")
