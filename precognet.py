from source.configs import SAVED_WEIGHTS_FOLDER, NON_FIGHT_VIDEOS_PATH, MAPPED_CLASSES
from source.model.model import model, load_model_weights, train_model
from source.video.frames import prepare_video
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np
import time
import os

def main():
    start_time = time.time()

    # train model
    precognet = model()
    features_test, labels_test = train_model(precognet)
    print("--- %.2f seconds ---" % (time.time() - start_time))

    # predict test set
    labels_predict = precognet.predict(features_test)
    labels_predict = np.argmax(labels_predict , axis=1)
    labels_test_normal = np.argmax(labels_test , axis=1)

    acc_score = accuracy_score(labels_predict, labels_test_normal)
    print (f'Test set accuracy is: {acc_score * 100}%')

def main2():
    start_time = time.time()
    # features_dataset, labels_dataset, _ = build_dataset()
    # _, features_test, _, labels_test = prepare_dataset_to_train(
    #     features_dataset, labels_dataset
    # )
    
    video_path = os.path.join(NON_FIGHT_VIDEOS_PATH, "7KXPekwe_0.avi")
    video = prepare_video(video_path)

    # train model
    precognet = model()
    load_model_weights(
        precognet,
        os.path.join(SAVED_WEIGHTS_FOLDER, 'precognet-0079d8f0-0aab.weights.h5')
    )

    # train_model(precognet)

    # show results
    for queued_frames in video:
        predicted_labels_probabilities = precognet.predict(
            np.expand_dims(queued_frames, axis = 0)
        )[0]

        predicted_label = np.argmax(predicted_labels_probabilities)
        probability = predicted_labels_probabilities.max()

        if probability < 0.6:
            predicted_label = -1

        print (f'Prediction: {MAPPED_CLASSES[predicted_label]} | Probability: {probability}')

if __name__ == '__main__':
    main()
