from typing import List
import cv2
import numpy as np

def extract_frames(file_path: str) -> List[np.ndarray]:
    """
    Extracts raw frames from a video file.

    Parameters:
    -----------
    file_path : str
        The file path to the video from which frames will be extracted.

    Returns:
    --------
    List[np.ndarray]
        A list of frames, where each frame is a numpy array representing 
        the image in BGR format (OpenCV's default color format).
    """
    images = []

    video_cap = cv2.VideoCapture(file_path)

    while True:
        ret, frame = video_cap.read()

        if not ret:
            break

        # Append the raw frame (BGR format) to the list
        images.append(frame)

    video_cap.release()

    return images
