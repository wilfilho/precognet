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
BATCH_SIZE = os.environ.get('BATCH_SIZE') # 16
EPOCHS = os.environ.get('EPOCHS') # 50

# Dataset sources
FIGHT_VIDEOS_PATH = os.environ.get('FIGHT_VIDEOS_PATH') # /Users/wilson/Documents/tcc/dataset/fight
NON_FIGHT_VIDEOS_PATH = os.environ.get('NON_FIGHT_VIDEOS_PATH') # /Users/wilson/Documents/tcc/dataset/nonfight
DATASET_LIMITATION =  int(os.environ.get('DATASET_LIMITATION')) # 20
SAVED_DATASETS_FOLDER = "/Users/wilson/Documents/tcc/precognet/.datasets"
SAVED_WEIGHTS_FOLDER = "/Users/wilson/Documents/tcc/precognet/.weights"
