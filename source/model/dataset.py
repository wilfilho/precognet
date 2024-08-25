import numpy as np
import os
from typing import Tuple, List
from source.configs import (
    MODEL_IMAGE_SIZE,
    FIGHT_VIDEOS_PATH,
    NON_FIGHT_VIDEOS_PATH,
    FIGHT_CLASS_NAME,
    NON_FIGHT_CLASS_NAME,
    DATASET_LIMITATION,
    SAVED_DATASETS_FOLDER
)
from source.model.normalizer import normalize_frames_to_model
from source.platform.folder import folder_walker
from source.platform.uuid import short_uuid
from source.video.frames import extract_frames
from source.video.preprocessing import resizer

def build_dataset() -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Build a dataset containing features, labels, and video file paths for both
    fight and non-fight classes.

    This function extracts video features, labels, and paths from fight and 
    non-fight video directories. It resizes and normalizes the frames and 
    returns a combined dataset with the corresponding labels and paths.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, List[str]]
        - features_dataset: Numpy array of features from fight and non-fight 
          videos.
        - labels_dataset: Numpy array of labels corresponding to the features.
        - videos_path: List of video file paths corresponding to the features.
    """
    fight_features, fight_labels, fight_videos_path = extract_features(
        FIGHT_VIDEOS_PATH, FIGHT_CLASS_NAME
    )
    
    nonfight_features, nonfight_labels, nonfight_videos_path = extract_features(
        NON_FIGHT_VIDEOS_PATH, NON_FIGHT_CLASS_NAME
    )
    
    features_dataset = np.asarray(fight_features + nonfight_features)
    labels_dataset = np.asarray(fight_labels + nonfight_labels)
    videos_path = fight_videos_path + nonfight_videos_path

    return features_dataset, labels_dataset, videos_path

def save_dataset(features_dataset: np.ndarray, 
                 labels_dataset: np.ndarray, 
                 videos_path: List[str]) -> None:
    """
    Save the dataset (features, labels, and video paths) as .npy files in a 
    new folder within the SAVED_DATASETS_FOLDER. Each dataset is assigned a 
    unique identifier.

    Parameters
    ----------
    features_dataset : np.ndarray
        The numpy array containing features extracted from videos.
    labels_dataset : np.ndarray
        The numpy array containing labels corresponding to the features.
    videos_path : List[str]
        A list of video file paths corresponding to the dataset.

    Returns
    -------
    None
    """
    uuid = short_uuid()

    if not os.path.exists(SAVED_DATASETS_FOLDER):
        os.mkdir(SAVED_DATASETS_FOLDER)

    new_dataset_folder = os.path.join(SAVED_DATASETS_FOLDER, uuid)
    os.mkdir(new_dataset_folder)

    features_file_name = f"features-{uuid}.npy"
    labels_file_name = f"labels-{uuid}.npy"
    videos_path_file_name = f"videos-path-{uuid}.npy"

    # Save the datasets as .npy files
    np.save(os.path.join(new_dataset_folder, features_file_name), 
            features_dataset)
    np.save(os.path.join(new_dataset_folder, labels_file_name), 
            labels_dataset)
    np.save(os.path.join(new_dataset_folder, videos_path_file_name), 
            videos_path)

def load_dataset(dataset_uuid: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load a saved dataset (features, labels, and video paths) from a specified 
    folder using the dataset UUID.

    Parameters
    ----------
    dataset_uuid : str
        The unique identifier for the dataset folder.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        A tuple containing:
        - features: Numpy array of the features dataset.
        - labels: Numpy array of the labels dataset.
        - videos_path: Numpy array of the video file paths.
    """
    dataset_folder = os.path.join(SAVED_DATASETS_FOLDER, dataset_uuid)

    features_file_name = os.path.join(
        dataset_folder, f"features-{dataset_uuid}.npy"
    )
    labels_file_name = os.path.join(
        dataset_folder, f"labels-{dataset_uuid}.npy"
    )
    videos_path_file_name = os.path.join(
        dataset_folder, f"videos-path-{dataset_uuid}.npy"
    )

    return (
        np.load(features_file_name),
        np.load(labels_file_name),
        np.load(videos_path_file_name)
    )


def extract_features(folder: str, class_name: str) -> Tuple[
        List[np.ndarray], List[str], List[str]]:
    """
    Extract features and labels from a folder of video files.

    This function traverses a folder, extracts frames from each video, resizes 
    them to the model input size, normalizes the frames, and generates 
    corresponding class labels.

    Parameters
    ----------
    folder : str
        The folder path containing video files.
    class_name : str
        The class name for the videos (e.g., 'fight' or 'non-fight').

    Returns
    -------
    Tuple[List[np.ndarray], List[str], List[str]]
        - features: List of numpy arrays, each representing video frames.
        - labels: List of labels corresponding to the videos.
        - videos_path: List of file paths for the processed videos.
    """
    if DATASET_LIMITATION is not None:
        videos = folder_walker(folder)[:DATASET_LIMITATION]
    else:
        videos = folder_walker(folder)

    features = []
    videos_path = []

    for video_path in videos:
        frames = normalize_frames_to_model(
            resizer(
                extract_frames(video_path), MODEL_IMAGE_SIZE
            )
        )
        features.append(frames)
        videos_path.append(video_path)

    labels = [class_name] * len(features)

    return features, labels, videos_path
