from source.model.dataset import (
    build_dataset,
    save_dataset,
    load_dataset,
    prepare_dataset_to_train
)
from source.model.model import model
from keras.api.callbacks import ReduceLROnPlateau, EarlyStopping
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
    print (len(features_dataset))
    print (videos_path)
    features_train, features_test, labels_train, labels_test = prepare_dataset_to_train(
        features_dataset, labels_dataset
    )
    # Create Early Stopping Callback to monitor the accuracy
    early_stopping_callback = EarlyStopping(
        monitor = 'val_accuracy',
        patience = 10,
        restore_best_weights = True
    )

    precognet = model()

    # Create ReduceLROnPlateau Callback to reduce overfitting by decreasing learning
    reduce_lr = ReduceLROnPlateau(monitor='val_loss',
                                                    factor=0.6,
                                                    patience=5,
                                                    min_lr=0.00005,
                                                    verbose=1)
    
    # Compiling the model 
    precognet.compile(loss = 'categorical_crossentropy', optimizer = 'sgd', metrics = ["accuracy"])
    
    # Fitting the model 
    precognet_model_history = precognet.fit(x = features_train, y = labels_train, epochs = 50, batch_size = 8 ,
                                                shuffle = True, validation_split = 0.2, callbacks = [early_stopping_callback,reduce_lr])
    print("--- %.2f seconds ---" % (time.time() - start_time))
    # model_evaluation_history = precognet.evaluate(features_test, labels_test)
    # plot_metric(precognet_model_history, 'loss', 'val_loss', 'Total Loss vs Total Validation Loss')
    plot_metric(precognet_model_history, 'accuracy', 'val_accuracy', 'Total Loss vs Total Validation Loss')
    # print(features_train.shape,labels_train.shape)
    # print(features_test.shape, labels_test.shape)
    # print (len(normalized_frames))
    # visible_frame = (normalized_frames*255).astype('uint8')
    # plt.imshow(visible_frame[70])
    # plt.show(block=True)

    # print(fight_videos[997])

def plot_metric(model_training_history, metric_name_1, metric_name_2, plot_name):
    
    metric_value_1 = model_training_history.history[metric_name_1]
    metric_value_2 = model_training_history.history[metric_name_2]
    
    # Get the Epochs Count
    epochs = range(len(metric_value_1))
 
    plt.plot(epochs, metric_value_1, 'blue', label = metric_name_1)
    plt.plot(epochs, metric_value_2, 'orange', label = metric_name_2)
 
    plt.title(str(plot_name))
 
    plt.legend()
    plt.show(block=True)

if __name__ == '__main__':
    main()
