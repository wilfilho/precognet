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
from source.configs import (
    SAVED_WEIGHTS_FOLDER,
    BATCH_SIZE, EPOCHS,
    SAVED_RESULTS_FOLDER,
    DATASET_FILE_NAME
)
from keras.api.utils import to_categorical
from source.platform.uuid import short_uuid
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from sklearn.model_selection import KFold
import json
import os
import time
import numpy as np
import h5py
import tensorflow as tf

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


def train_model():
    """
    Trains the provided model on the given dataset using early stopping and
    learning rate reduction callbacks.

    Args:
        model_to_train (Any): The model to be trained.
        save (bool, optional): Whether to save the trained model. Defaults to True.

    Returns:
        None
    """
    with h5py.File(DATASET_FILE_NAME, 'r') as base_dataset:
        print ("Loading dataset...")
        dataset_features = np.array(base_dataset.get('fight-features')) + np.array(base_dataset.get('non-fight-features'))
        print ("Loading labels...")
        dataset_labels = np.array(base_dataset.get('fight-labels')) + np.array(base_dataset.get('non-fight-labels'))
        
        print ("Configuring model constraints...")
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
        
        print ("Compiling model...")
        strategy = tf.distribute.MirroredStrategy()
        with strategy.scope(): 
            model_pipeline = model()
            model_pipeline.compile(
                loss="categorical_crossentropy",
                optimizer="adam",
                metrics=["accuracy"]
            )
        
        kf = KFold(n_splits=5, shuffle=True, random_state=42)

        predictions = []
        true_labels = []
        scores = []
        training_times = []

        start_time = time.time()

        print ("Pre training model...")
        for train_index, test_index in kf.split(dataset_features):
            X_train, X_test = dataset_features[train_index], dataset_features[test_index]
            y_train, y_test = to_categorical(dataset_labels[train_index]), to_categorical(dataset_labels[test_index])

            print(f'Train set: {len(train_index)}, Test set:{len(test_index)}')
            internal_start_time = time.time()
            model_pipeline.fit(
                x=X_train,
                y=y_train,
                epochs=EPOCHS,
                batch_size=BATCH_SIZE,
                shuffle=True,
                validation_split=0.2,
                callbacks=[early_stopping_callback, reduce_lr]
            )

            y_pred = model_pipeline.predict(X_test)
            y_pred=np.argmax(y_pred, axis=1)
            y_test=np.argmax(y_test, axis=1)
            predictions.extend(y_pred)
            true_labels.extend(y_test)

            t_result = time.time() - internal_start_time

            print ("Total training time: {:.2f}".format(t_result))
            training_times.append(t_result)

            scores.append(accuracy_score(y_test, y_pred))
        
        training_time = time.time() - start_time

        recall = recall_score(true_labels, predictions, average='macro')
        precision = precision_score(true_labels, predictions, average='macro')
        f1 = f1_score(true_labels, predictions, average='macro')
        accuracy = accuracy_score(true_labels, predictions)
        cm = confusion_matrix(true_labels, predictions)

        return {
            "model": model_pipeline,
            "recall": recall,
            "precision": precision,
            "f1": f1,
            "accuracy": accuracy,
            "cm": cm,
            "training_time": training_time,
            "k_fold_training_times": training_times
        }

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
