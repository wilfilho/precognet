from source.model.dataset import (
    build_dataset,
    save_dataset,
    load_dataset,
    prepare_dataset_to_train
)
from source.configs import SAVED_WEIGHTS_FOLDER
from source.model.model import model, train_model, load_model_weights
from source.platform.uuid import short_uuid
from keras.api.callbacks import ReduceLROnPlateau, EarlyStopping
from keras.api.models import load_model
import matplotlib.pyplot as plt
import numpy as np
import time
import os
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

def main():
    start_time = time.time()
    features_dataset, labels_dataset, _ = build_dataset()
    _, features_test, _, labels_test = (
        prepare_dataset_to_train(features_dataset, labels_dataset)
    )

    # train model
    precognet = model()
    load_model_weights(
        precognet,
        os.path.join(SAVED_WEIGHTS_FOLDER, 'precognet-0079d8f0-0aab.weights.h5')
    )

    # train_model(precognet)

    # show results
    print("--- %.2f seconds ---" % (time.time() - start_time))
    labels_predict = precognet.predict(features_test)
    labels_predict = np.argmax(labels_predict , axis=1)
    labels_test_normal = np.argmax(labels_test , axis=1)
    score_on_testing = accuracy_score(labels_predict, labels_test_normal)
    report = classification_report(labels_test_normal,labels_predict)
    print(f'Accuracy Score is : {score_on_testing}')
    print(f'Classification Report is : \n${report}')

if __name__ == '__main__':
    main()
