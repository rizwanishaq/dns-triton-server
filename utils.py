import time
import numpy as np


def unique_sequence_id() -> np.uint64:
    """
    Generates a unique sequence ID based on the current time.

    Returns:
        np.uint64: A unique sequence ID.
    """
    return np.uint64(time.time() * 1000000000)
