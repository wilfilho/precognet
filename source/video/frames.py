from typing import List
import cv2
import numpy as np
from source.model.normalizer import normalize_frames_to_model
from source.video.preprocessing import resizer
from source.configs import MODEL_IMAGE_SIZE

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

def prepare_video(video_path):
    features = []
    frames = normalize_frames_to_model(
        resizer(
            extract_frames(video_path), MODEL_IMAGE_SIZE
        )
    )

    # split all frames into sub lists of 16 items
    # removes the last item that doesn't have 16 items
    all_subframes = [frames[i:i + 16] for i in range(0, len(frames), 16)][:-1]

    for subframe in all_subframes:
        features.append(subframe)
    
    return features