import os

# Model configs & classes
MODEL_IMAGE_SIZE = (224, 224)
FIGHT_CLASS_NAME = 0
NON_FIGHT_CLASS_NAME = 1
UNKNOWN_CLASS_NAME = -1
MAPPED_CLASSES = {
    FIGHT_CLASS_NAME: "fight",
    NON_FIGHT_CLASS_NAME: "non fight",
    UNKNOWN_CLASS_NAME: "unknown"
}
BATCH_SIZE = int(os.environ.get('BATCH_SIZE')) # 16
EPOCHS = int(os.environ.get('EPOCHS')) # 50

# Dataset sources
FIGHT_VIDEOS_PATH = os.environ.get('FIGHT_VIDEOS_PATH') # /Users/wilson/Documents/tcc/dataset/fight
NON_FIGHT_VIDEOS_PATH = os.environ.get('NON_FIGHT_VIDEOS_PATH') # /Users/wilson/Documents/tcc/dataset/nonfight
DATASET_LIMITATION =  int(os.environ.get('DATASET_LIMITATION')) # 20
SAVED_DATASETS_FOLDER = os.environ.get('SAVED_DATASETS_FOLDER') # /Users/wilson/Documents/tcc/precognet/.datasets
SAVED_WEIGHTS_FOLDER = os.environ.get('SAVED_WEIGHTS_FOLDER') # /Users/wilson/Documents/tcc/precognet/.weights
SAVED_RESULTS_FOLDER = os.environ.get('SAVED_RESULTS_FOLDER') # /Users/wilson/Documents/tcc/precognet/.results
RAW_DATASET_URL = "https://drive.google.com/file/d/1vPqeCJ45u8U0v1W5OVCHpJnoWMev5fQ3"
DATASET_FILE_NAME = "precognet-dataset.h5"
