import tensorflow as tf
from keras._tf_keras.keras.preprocessing.image import ImageDataGenerator
from keras._tf_keras.keras.models import Sequential
from keras._tf_keras.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from keras._tf_keras.keras.applications import MobileNetV2
from keras._tf_keras.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras._tf_keras.keras.regularizers import l2
import time
import pickle
import os

def create_generators(train_images, test_images):
    """Create and return the training and testing data generators."""
    train_gen = ImageDataGenerator(
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        rotation_range=20,
        brightness_range=(0.8, 1.2),
        width_shift_range=0.2,
        height_shift_range=0.2,
        rescale=1./255, 
        validation_split=0.2
    )

    test_gen = ImageDataGenerator(rescale=1./255)

    training_data = train_gen.flow_from_directory(
        train_images,
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical',
        shuffle=True,
        subset='training'  # Use this subset for training
    )

    testing_data = test_gen.flow_from_directory(
        test_images,
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical'
    )

    validation_data = train_gen.flow_from_directory(
        train_images,
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical',
        shuffle=True,
        subset="validation"
    )

    return training_data, testing_data, validation_data

def create_class_mapping(training_data):
    """Create the mapping between numeric labels and class names."""
    Train_class = training_data.class_indices
    Result_class = {value_tag: face_tag for face_tag, value_tag in Train_class.items()}

    with open(r"ResultMap.pkl", "wb") as Final_mapping:
        pickle.dump(Result_class, Final_mapping)

    print("Mapping of face and its numerical value", Result_class)
    return Result_class

def create_model(output_neurons):
    """Create and compile a model using MobileNetV2 as the base."""
    base_model = MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
    base_model.trainable = False 

    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dense(256, activation='relu', kernel_regularizer=l2(0.01)),
        Dropout(0.5),
        Dense(128, activation='relu', kernel_regularizer=l2(0.01)),
        Dropout(0.5),
        Dense(64, activation='relu', kernel_regularizer=l2(0.01)),
        Dropout(0.5),
        Dense(32, activation='relu', kernel_regularizer=l2(0.01)),
        Dropout(0.5),
        Dense(output_neurons, activation='softmax')
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model, base_model

def train_model(model, training_data, validation_data):
    """Train the model."""
    callbacks = [
        EarlyStopping(patience=5, verbose=1, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1)
    ]

    StartTime = time.time()
    model.fit(
        training_data,
        epochs=100,
        validation_data=validation_data,
        callbacks=callbacks
    )
    EndTime = time.time()

    print('Total Training Time taken: ', round((EndTime - StartTime) / 60), 'Minutes')

def fine_tune_model(model, base_model, training_data, validation_data, fine_tune_at=100):
    base_model.trainable = True
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    callbacks = [
        EarlyStopping(patience=5, verbose=1, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6, verbose=1)
    ]

    model.fit(
        training_data,
        epochs=30,
        validation_data=validation_data,
        callbacks=callbacks
    )

def load_model(model_path):
    """Load a saved model from a file."""
    if os.path.exists(model_path):
        model = tf.keras.models.load_model(model_path)
        print(f"Model loaded successfully from {model_path}.")
        return model
    else:
        raise FileNotFoundError(f"No model found at {model_path}")
