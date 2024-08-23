from source.configs import (
    MODEL_IMAGE_SIZE,
    FIGHT_VIDEOS_PATH,
    NON_FIGHT_VIDEOS_PATH
)
from source.model.model import model
from source.model.normalizer import normalize_frames_to_model
from source.platform.folder import folder_walker
from source.video.frames import extract_frames
from source.video.preprocessing import resizer
import matplotlib.pyplot as plt
import time

def main():
    # precog_model = model()
    # precog_model.summary()
    fight_videos = folder_walker(FIGHT_VIDEOS_PATH)
    nonfight_videos = folder_walker(NON_FIGHT_VIDEOS_PATH)
    # start_time = time.time()
    # frames = resizer(extract_frames(selected_video), MODEL_IMAGE_SIZE)
    # normalized_frames = normalize_frames_to_model(frames)
    # print("--- %.2f seconds ---" % (time.time() - start_time))
    # print (len(normalized_frames))
    # visible_frame = (normalized_frames*255).astype('uint8')
    # plt.imshow(visible_frame[70])
    # plt.show(block=True)

    # print(fight_videos[997])

if __name__ == '__main__':
    main()
