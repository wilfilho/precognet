from keras.src.models import Sequential
from keras.src.layers import *
from keras.src.applications.mobilenet_v2 import MobileNetV2

def model():
    mobilenet = MobileNetV2(include_top=False, weights="imagenet", input_shape=(224, 224, 3))
    model = Sequential()
    model.add(Input(shape=(16, 224, 224, 3)))
    model.add(TimeDistributed(mobilenet))
    model.add(ConvLSTM2D(filters=64, kernel_size=(3, 3), activation='tanh'))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    
    return model
