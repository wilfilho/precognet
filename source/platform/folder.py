from typing import List
import os

def folder_walker(folder_path: str) -> List[str]:
    """
    Walk through all files in a given folder and return their full file paths.

    Parameters
    ----------
    folder_path : str
        The path to the folder to be traversed.

    Returns
    -------
    List[str]
        A list of file paths for all the files found within the 
        folder and its subdirectories.
    """
    return [
        os.path.join(root, file) 
        for root, _, files in os.walk(folder_path) 
        for file in files
    ]