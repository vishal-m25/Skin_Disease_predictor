import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
import os

def create_cnn_model(input_shape, num_classes):
    model = models.Sequential()
    model.add(layers.Conv2D(8, kernel_size=(3,3), activation='relu', input_shape=input_shape))
    model.add(layers.Conv2D(16, kernel_size=(3,3), activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Dropout(0.25))
    model.add(layers.Conv2D(32, kernel_size=(3, 3), activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Conv2D(32, kernel_size=(3, 3), activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Dropout(0.25))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(num_classes, activation='softmax'))
    return model

train_data_dir = 'train'
test_data_dir = 'test'
img_height, img_width = 58, 58
batch_size = 10
epochs = 40


def train_cnn_model(train_data_dir, test_data_dir, img_height, img_width, batch_size, epochs):
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True
    )

    test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='categorical'
    )

    test_generator = test_datagen.flow_from_directory(
        test_data_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='categorical'
    )

    num_classes = len(train_generator.class_indices)

    cnn_model = create_cnn_model(input_shape=(img_height, img_width, 3), num_classes=num_classes)
    cnn_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    cnn_model.fit(train_generator, epochs=epochs, validation_data=test_generator)

    test_loss, test_acc = cnn_model.evaluate(test_generator)
    print(f"Test accuracy: {test_acc}")

    return cnn_model  
trained_model = train_cnn_model(train_data_dir, test_data_dir, img_height, img_width, batch_size, epochs)


trained_model.save("model.h5")
