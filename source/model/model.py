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
from source.configs import SAVED_WEIGHTS_FOLDER, BATCH_SIZE
from source.platform.uuid import short_uuid
import os

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
        optimizer='sgd',
        metrics=["accuracy"]
    )
    
    model_to_train.fit(
        x=features_train,
        y=labels_train,
        epochs=50,
        batch_size=BATCH_SIZE,
        shuffle=True,
        validation_split=0.2,
        callbacks=[early_stopping_callback, reduce_lr]
    )

    if save:
        Path(SAVED_WEIGHTS_FOLDER).mkdir(exist_ok=True)
        weights = f'precognet-{short_uuid()}.weights.h5'
        folder_to_save = os.path.join(SAVED_WEIGHTS_FOLDER, weights)
        model_to_train.save_weights(folder_to_save)
    
    return features_test, labels_test

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