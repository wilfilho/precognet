from source.model.model import train_model
from source.model.dataset import extract_features, save_dataset
from source.configs import (
    FIGHT_VIDEOS_PATH, 
    FIGHT_CLASS_NAME, 
    NON_FIGHT_VIDEOS_PATH, 
    NON_FIGHT_CLASS_NAME,
    DATASET_FILE_NAME
)
import numpy as np
import h5py


def main():
    print (train_model())

def main():
    print ('Loading videos...')
    fight_features, fight_labels, _ = extract_features(
        FIGHT_VIDEOS_PATH, FIGHT_CLASS_NAME
    )
    non_fight_features, non_fight_labels, _ = extract_features(
        NON_FIGHT_VIDEOS_PATH, NON_FIGHT_CLASS_NAME
    )
    print ('Converting into numpy arrays')
    f_features_dataset = np.asarray(fight_features)
    f_labels_dataset = np.array(fight_labels)
    non_f_features_dataset = np.asarray(non_fight_features)
    non_f_labels_dataset = np.array(non_fight_labels)

    print ('Saving into file')
    with h5py.File(DATASET_FILE_NAME, 'w') as base:
        base.create_dataset('fight-features', data = f_features_dataset)
        base.create_dataset('fight-labels', data = f_labels_dataset)
        base.create_dataset('non-fight-features', data = non_f_features_dataset)
        base.create_dataset('non-fight-labels', data = non_f_labels_dataset)
    
    print ('Finished')
    

# def main2():
#     start_time = time.time()
#     # features_dataset, labels_dataset, _ = build_dataset()
#     # _, features_test, _, labels_test = prepare_dataset_to_train(
#     #     features_dataset, labels_dataset
#     # )
    
#     video_path = os.path.join(NON_FIGHT_VIDEOS_PATH, "7KXPekwe_0.avi")
#     video = prepare_video(video_path)

#     # train model
#     precognet = model()
#     load_model_weights(
#         precognet,
#         os.path.join(SAVED_WEIGHTS_FOLDER, 'precognet-0079d8f0-0aab.weights.h5')
#     )

#     # train_model(precognet)

#     # show results
#     for queued_frames in video:
#         predicted_labels_probabilities = precognet.predict(
#             np.expand_dims(queued_frames, axis = 0)
#         )[0]

#         predicted_label = np.argmax(predicted_labels_probabilities)
#         probability = predicted_labels_probabilities.max()

#         if probability < 0.6:
#             predicted_label = -1

#         print (f'Prediction: {MAPPED_CLASSES[predicted_label]} | Probability: {probability}')

if __name__ == '__main__':
    main()
