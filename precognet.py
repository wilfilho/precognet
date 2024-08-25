from source.model.dataset import build_dataset
import matplotlib.pyplot as plt
import numpy as np
import time

FIGHT_CLASS_NAME = "fight"
NON_FIGHT_CLASS_NAME = "nonfight"

def main():
    # precog_model = model()
    # precog_model.summary()
    # fight_videos = folder_walker(FIGHT_VIDEOS_PATH)[:3]
    # nonfight_videos = folder_walker(NON_FIGHT_VIDEOS_PATH)[:3]
    # all_videos = fight_videos + nonfight_videos
    # print (all_videos)
    start_time = time.time()
    # frames = resizer(extract_frames(selected_video), MODEL_IMAGE_SIZE)
    # normalized_frames = normalize_frames_to_model(frames)
    features_dataset, labels_dataset, videos_path = build_dataset()
    print("--- %.2f seconds ---" % (time.time() - start_time))
    print (len(features_dataset))
    print (len(labels_dataset))
    print (videos_path)
    # print (len(normalized_frames))
    # visible_frame = (normalized_frames*255).astype('uint8')
    # plt.imshow(visible_frame[70])
    # plt.show(block=True)

    # print(fight_videos[997])


if __name__ == '__main__':
    main()
