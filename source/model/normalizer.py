import numpy as np
from typing import List

def normalize_frames_to_model(frames: List[np.ndarray]) -> np.ndarray:
    """
    Normalize a list of video frames to the range [0, 1] and convert
    to float16 format.

    Parameters
    ----------
    frames : List[np.ndarray]
        A list of video frames, where each frame is represented
        as a numpy array.

    Returns
    -------
    np.ndarray
        A numpy array of normalized frames, where pixel values are scaled
        to the range [0, 1] and the data type is cast to float16.
    """
    return (np.array(frames) / 255.0).astype(np.float16)