import cv2
import numpy as np
from typing import List, Tuple

def resizer(frames: List[np.ndarray], new_size: Tuple[int, int]) -> List[np.ndarray]:
    """
    Resize and convert a list of video frames to RGB format.

    Parameters
    ----------
    frames : List[np.ndarray]
        A list of frames (in BGR format), where each frame is a numpy array.
    new_size : Tuple[int, int]
        The desired size for resizing each frame, given as (width, height).

    Returns
    -------
    List[np.ndarray]
        A list of resized frames, where each frame is converted to RGB format
        and resized to the specified size.
    """
    resized_frames = []

    for frame in frames:
        # Convert the frame from BGR (OpenCV default) to RGB
        rgb_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        frame_resized = cv2.resize(
            rgb_img, 
            dsize=new_size,
            interpolation=cv2.INTER_CUBIC
        )

        resized_frames.append(frame_resized)
    
    return resized_frames
