import tensorflow as tf
from keras._tf_keras.keras.preprocessing.image import ImageDataGenerator
import pickle
from keras._tf_keras.keras.models import Sequential
from keras._tf_keras.keras.layers import Conv2D, MaxPool2D, Flatten, Dense, BatchNormalization, Dropout
from keras._tf_keras.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras._tf_keras.keras.regularizers import l2
import time
import numpy as np
from keras._tf_keras.keras.preprocessing import image
import os


def create_generators(train_images, test_images):
    """Create and return the training and testing data generators."""
    train_gen = ImageDataGenerator(
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        rotation_range=20,
        brightness_range=(0.7, 1.3),
        width_shift_range= 0.2,
        height_shift_range= 0.2
    )

    test_gen = ImageDataGenerator()

    training_data = train_gen.flow_from_directory(
        train_images, 
        target_size = (100,100),
        batch_size = 30,
        class_mode = 'categorical',
        shuffle=True
    )

    testing_data = test_gen.flow_from_directory(
        test_images,
        target_size=(100, 100),
        batch_size=30,
        class_mode='categorical',

    )
    
    return training_data, testing_data


def create_class_mapping(training_data):
    """Create the mapping between numeric labels and class names."""
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

    model.add(Conv2D(16, kernel_size=(5, 5), strides=(1, 1), input_shape=(100, 100, 3), activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))

    model.add(Conv2D(32, kernel_size=(3, 3), strides=(1, 1), activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))

    model.add(Flatten())

    model.add(Dense(64, activation='relu'))
    model.add(Dense(output_neurons, activation='softmax'))

    model.compile(
        loss='categorical_crossentropy',
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        metrics=['Accuracy']
    )


    return model


def train_model(model, training_data, testing_data):
    """Train the CNN model."""
    callbacks = EarlyStopping(
        patience=5, 
        verbose=1, 
        min_delta=0.005
    )

    # Reduce learning rate if validation loss plateaus
    lr_scheduler = ReduceLROnPlateau(
        monitor='val_loss', 
        factor=0.5, 
        patience=3, 
        min_lr=1e-6,
        verbose = 1

    )

    StartTime = time.time()
    model.fit(
        training_data,
        epochs=100,
        validation_data=testing_data,
        callbacks= [callbacks, lr_scheduler]
    )
    EndTime = time.time()

    print('Total Training Time taken: ', round((EndTime - StartTime) / 60), 'Minutes')

def load_model(model_path):
    """Load a saved model from a file."""
    if os.path.exists(model_path):
        model = tf.keras.models.load_model(model_path)
        print(f"Model loaded successfully from {model_path}.")
        return model
    else:
        raise FileNotFoundError(f"No model found at {model_path}")