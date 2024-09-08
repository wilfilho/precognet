from typing import Any
from keras.api.callbacks import ReduceLROnPlateau, EarlyStopping
from keras.src.models import Sequential
from keras.src.layers import *
from keras.src.applications.mobilenet_v2 import MobileNetV2
from pathlib import Path
from source.model.dataset import (
    build_dataset,
    save_dataset,
    load_dataset,
    prepare_dataset_to_train
)
from source.configs import SAVED_WEIGHTS_FOLDER, BATCH_SIZE, EPOCHS, SAVED_RESULTS_FOLDER
from source.platform.uuid import short_uuid
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
import json
import os
import time
import numpy as np

def model() -> Sequential:
    """
    Constructs a sequential model using a pre-trained MobileNetV2 as the base 
    and a ConvLSTM2D layer for sequence processing. It is followed by fully 
    connected layers for classification.

    Returns:
        Sequential: A Keras Sequential model object.
    """
    mobilenet = MobileNetV2(
        include_top=False, weights="imagenet", input_shape=(224, 224, 3)
    )
    
    model = Sequential()
    model.add(Input(shape=(16, 224, 224, 3)))
    model.add(TimeDistributed(mobilenet))
    model.add(ConvLSTM2D(filters=64, kernel_size=(3, 3), activation='tanh'))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2, activation='softmax'))
    
    return model


def train_model(model_to_train: Sequential, save: bool = True):
    """
    Trains the provided model on the given dataset using early stopping and
    learning rate reduction callbacks.

    Args:
        model_to_train (Any): The model to be trained.
        save (bool, optional): Whether to save the trained model. Defaults to True.

    Returns:
        None
    """
    features_dataset, labels_dataset, _ = build_dataset()
    
    features_train, features_test, labels_train, labels_test = (
        prepare_dataset_to_train(features_dataset, labels_dataset)
    )
    
    early_stopping_callback = EarlyStopping(
        monitor='val_accuracy',
        patience=10,
        restore_best_weights=True
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.6,
        patience=5,
        min_lr=0.00005,
        verbose=1
    )
    
    model_to_train.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=["accuracy"]
    )
    
    model_to_train.fit(
        x=features_train,
        y=labels_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        shuffle=True,
        validation_split=0.2,
        callbacks=[early_stopping_callback, reduce_lr]
    )

    if save:
        save_dataset(features_test, labels_test, [])
        Path(SAVED_WEIGHTS_FOLDER).mkdir(exist_ok=True)
        weight_uuid = short_uuid()
        weights = f'precognet-{weight_uuid}.weights.h5'
        folder_to_save = os.path.join(SAVED_WEIGHTS_FOLDER, weights)
        model_to_train.save_weights(folder_to_save)
        return features_test, labels_test, folder_to_save, weights, weight_uuid
    
    return features_test, labels_test, None, None

def load_model_weights(internal_model: Sequential, weights_path: str) -> None:
    """
    Loads the weights from a specified file into the given model.

    Args:
        internal_model (Sequential): The Keras Sequential model into which 
                                     the weights will be loaded.
        weights_path (str): The file path where the model weights are stored.

    Returns:
        None
    """
    internal_model.load_weights(weights_path)

def train_and_compile_results():
    start_time = time.time()

    print ("[1] The training has been started.")
    # train model
    precognet = model()
    features_test, labels_test, folder_saved, file_saved, weight_uuid = train_model(precognet)
    seconds_to_train = time.time() - start_time
    print (f'[2] The training has been completed in {seconds_to_train} seconds.')

    # predict test set
    labels_predict = precognet.predict(features_test)
    labels_predict = np.argmax(labels_predict , axis=1)
    labels_test_normal = np.argmax(labels_test , axis=1)

    acc_score_final = accuracy_score(labels_predict, labels_test_normal)
    f1_score_final = f1_score(labels_predict, labels_test_normal)
    precision_score_final = precision_score(labels_predict, labels_test_normal)
    recall_score_final = recall_score(labels_predict, labels_test_normal)
    print ("[3] The results have been captured.")

    result = json.dumps({
        "seconds_to_train": seconds_to_train,
        "saved_weights_path": folder_saved,
        "saved_weights_filename": file_saved,
        "metrics": {
            "accuracy": acc_score_final,
            "f1_score": f1_score_final,
            "precision": precision_score_final,
            "recall": recall_score_final
        }
    })

    file_to_save = f'result-precognet-{time.time()}-{weight_uuid}.json'
    results_to_save_path = os.path.join(SAVED_RESULTS_FOLDER, file_to_save)

    Path(SAVED_RESULTS_FOLDER).mkdir(exist_ok=True) 
    
    with open(results_to_save_path, 'w') as f:
        f.write(result)
    
    print ("[4] Training and testing has been completed.")
