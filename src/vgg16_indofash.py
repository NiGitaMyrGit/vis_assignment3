#!/usr/bin/env python3
#import packages
import os
# tf tools
import tensorflow as tf
# image processsing
from tensorflow.keras.preprocessing.image import (load_img, img_to_array, ImageDataGenerator)
# VGG16 mode and preproccessing
from tensorflow.keras.applications.vgg16 import (preprocess_input, VGG16, decode_predictions)
# layers
from tensorflow.keras.layers import (Flatten, Dense, Dropout, BatchNormalization)
# generic model object
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
# optimizers
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.optimizers import SGD
#scikit-learn
from sklearn.metrics import classification_report
#plotting
import numpy as np
import matplotlib.pyplot as plt
#load in labels
import pandas as pd

#help function to plot and save it too
def plot_history(H, epochs, plot_name):
    outpath = os.path.join("out", f"{plot_name}.png")
    plt.style.use("seaborn-colorblind")
    plt.figure(figsize=(12,6))
    plt.suptitle(f"History for Indo-fashion dataset trained on VGG16", fontsize=14)
    plt.subplot(1,2,1)
    plt.plot(np.arange(0, epochs), H.history["loss"], label="train")
    plt.plot(np.arange(0, epochs), H.history["val_loss"], label="val_labels", linestyle=":")
    plt.title("Loss curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.tight_layout()
    plt.legend()
    plt.subplot(1,2,2)
    plt.plot(np.arange(0, epochs), H.history["accuracy"], label="train")
    plt.plot(np.arange(0, epochs), H.history["val_acc"], label="val_labels", linestyle=":")
    plt.title("Accuracy curve")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.tight_layout()
    plt.legend()
    plt.savefig(os.path.join(outpath))


#function for saving classification report
def save_report(report, report_name, learning_rate, epochs, batch_size):
    outpath = os.path.join("out", f"{report_name}.txt")
    with open(outpath,"w") as file:
        file.write(f"Classification report\nData: Indo_fashion\nModel: VGG16\nEpochs: {epochs}\nLearning rate: {learning_rate}\nBatch size: {batch_size}\n")
        file.write(str(report)) 


def run():
    #get labels
    test_labels = pd.read_json(os.path.join("in","test_data.json"), lines=True)
    train_labels = pd.read_json(os.path.join("in","train_data.json"), lines=True)
    val_labels = pd.read_json(os.path.join("in","val_data.json"), lines=True)
    test_df = pd.DataFrame(test_labels)
    train_df = pd.DataFrame(train_labels)
    val_df  = pd.DataFrame(val_labels)
    train_df['image_path'] = "in/" + train_df['image_path']
    test_df['image_path'] = "in/" + test_df['image_path']
    val_df['image_path'] = "in/" + val_df['image_path']
    #amount of labels
    n_classes = 15
    img_rows = 224
    img_cols = 224
    batch_size = 128
    n_testsamples = 75 #TODO there are 7500 all in all, trying on a subset
    n_train_samples = 911 #TODO there are 91166, tryin on subset

    #data augmentation
    train_datagen = ImageDataGenerator(rescale=1. / 255,
                                       rotation_range=40,
                                       width_shift_range=0.2,
                                       height_shift_range=0.2,
                                       shear_range=0.2,
                                       zoom_range=0.2,
                                       horizontal_flip=True,
                                       validation_split=0.2,
                                       fill_mode='nearest')
    test_datagen = ImageDataGenerator(rescale=1. / 255)
    val_datagen = ImageDataGenerator(rescale=1. / 255)

    # adding directory path to pd dataframe
    
    print("hello")
    #get data from directories
    train_generator = train_datagen.flow_from_dataframe(dataframe=train_df, 
                                                        x_col="image_path",
                                                        y_col="class_label",
                                                        target_size=(img_rows, img_cols),
                                                        batch_size=batch_size,
                                                        class_mode='categorical',
                                                        shuffle=True)
    print(train_generator)
    test_generator = test_datagen.flow_from_dataframe(dataframe=test_df, 
                                                            x_col="image_path",
                                                            y_col="class_label",
                                                            target_size=(img_rows, img_cols),
                                                            batch_size=batch_size,
                                                            class_mode='categorical',
                                                            shuffle = False)
    print(test_generator)
    validation_generator = val_datagen.flow_from_dataframe(dataframe=val_df, 
                                                            x_col="image_path",
                                                            y_col="class_label",
                                                            target_size=(img_rows, img_cols),
                                                            batch_size=batch_size,
                                                            class_mode='categorical',
                                                            shuffle = False)
    print(validation_generator)
    #load model
    base_model = VGG16(weights='imagenet', 
                include_top=False,
                pooling='avg',
                input_shape=(224, 224, 3))
    print("disable layers")
    # disable convolutional layers
    for layer in base_model.layers:
        layer.trainable = False
    # clear session if model is run multiple times
    tf.keras.backend.clear_session()
    # add new classifier layers
    flat1 = Flatten()(base_model.layers[-1].output)
    x = Dense(256, activation='relu')(flat1)
    x = Dense(128, activation='relu')(x)
    x = Dense(64, activation='relu')(x)
    x = Dense(32, activation='relu')(x)
    outputs = Dense(15, activation='softmax')(x)
    # define new model
    model = Model(inputs=base_model.inputs, 
                outputs=outputs)
    # compile
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=0.01,
    decay_steps=10000,
    decay_rate=0.9)
    sgd = SGD(learning_rate=lr_schedule)
    print("model compile")
    model.compile(optimizer=sgd,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'],
                  run_eagerly=True)

    # stop training if validation loss i 0 for 2 epochs
    callbacks  = [EarlyStopping(monitor='val_loss',
                                   min_delta=0,
                                   patience=2, #TODO set to 2
                                   mode='auto')]

    H = model.fit_generator(train_generator,
                            steps_per_epoch=n_train_samples//batch_size,
                            epochs = 3, #TODO set to 5
                            validation_data=validation_generator,
                            validation_steps=n_testsamples//batch_size,
                            callbacks = callbacks)

    print("validation_generator resets")              
    validation_generator.reset()
    print("validating model")
    # evaluate model
    Y_pred = model.predict(validation_generator, steps=len(test_generator))
    y_pred = np.argmax(Y_pred, axis=1)
    print('Classification Report')
    # get labels
    target_names = list(train_generator.class_indices.keys())
    print("classification report")
    # create classification report
    report = classification_report(test_generator.classes, 
                                   y_pred, 
                                   target_names=target_names)
    # get actual number of epochs used
    n_epochs = len(H.history['loss'])
    # save report
    save_report(report, "classification_report", 0.01,3,128)
    # save history plot
    plot_history(H, n_epochs, "history plot")

    return print(report)


if __name__=="__main__":
    run()