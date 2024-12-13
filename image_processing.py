import tensorflow as tf
from keras._tf_keras.keras.preprocessing.image import ImageDataGenerator
import pickle
from keras._tf_keras.keras.models import Sequential
from keras._tf_keras.keras.layers import Conv2D, MaxPool2D, Flatten, Dense, BatchNormalization, Dropout
from keras._tf_keras.keras.callbacks import EarlyStopping
from keras._tf_keras.keras.regularizers import l2
import time
import numpy as np
from keras._tf_keras.keras.preprocessing import image
import os
import glob


def create_generators(train_images, test_images):
    """Create and return the training and testing data generators."""
    train_gen = ImageDataGenerator(
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        rotation_range=15,
        brightness_range=(0.8, 1.2),
    )

    test_gen = ImageDataGenerator()

    training_data = train_gen.flow_from_directory(
        train_images, 
        target_size = (100,100),
        batch_size = 30,
        class_mode = 'categorical',
    )

    testing_data = test_gen.flow_from_directory(
        test_images,
        target_size=(100, 100),
        batch_size=30,
        class_mode='categorical',
    )
    
    return training_data, testing_data


def create_class_mapping(training_data, testing_data):
    """Create the mapping between numeric labels and class names."""
    testing_data.class_indices
    Train_class = training_data.class_indices
    Result_class = {}
    for value_tag, face_tag in zip(Train_class.values(),Train_class.keys()):
        Result_class[value_tag] = face_tag

    with open(r"detect_faces\ResultMap.pkl", "wb") as Final_mapping:
        pickle.dump(Result_class, Final_mapping)

    print("Mapping of face and its numerical value", Result_class)
    return Result_class


def create_model(output_neurons):
    """Create and compile the enhanced CNN model."""
    model = Sequential()

    model.add(Conv2D(32, kernel_size=(3, 3), strides=(1, 1), input_shape=(100, 100, 3), activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(BatchNormalization())

    model.add(Conv2D(64, kernel_size=(3, 3), strides=(1, 1), activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(BatchNormalization())

    model.add(Conv2D(128, kernel_size=(3, 3), strides=(1, 1), activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(BatchNormalization())

    model.add(Flatten())
    model.add(Dense(128, activation='relu',  kernel_regularizer = l2(0.001)))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu',  kernel_regularizer = l2(0.001)))
    model.add(Dropout(0.5))
    model.add(Dense(16, activation='relu',  kernel_regularizer = l2(0.001)))
    model.add(Dropout(0.5))
    model.add(Dense(output_neurons, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model



def train_model(model, training_data, testing_data):
    """Train the CNN model."""
    # callbacks = EarlyStopping(
    #                 min_delta=0.005,
    #                 patience=5,
    #                  verbose=1
    #                 )

    StartTime = time.time()
    model.fit(
        training_data,
        epochs=100,
        validation_data=testing_data,
        # callbacks=callbacks
    )
    EndTime = time.time()

    print('Total Training Time taken: ', round((EndTime - StartTime) / 60), 'Minutes')