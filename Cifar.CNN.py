
import numpy as np
import pandas as pd
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.datasets import cifar10
from pathlib import Path
from PIL import Image
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import RandomFlip, RandomRotation, RandomContrast, RandomZoom
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
from numba import jit, cuda

from keras.utils import to_categorical
from tensorflow.keras.utils import image_dataset_from_directory
from tensorflow.keras import regularizers
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

# JIT compile a simple function
@jit
def my_function(x):
    return x + 1

result = my_function(10)
print(result)
#definig the data direction
dataset_dir = r'D:\Final project 3\dataset' #adjust your direction as you put your dataset file

# Assuming you have separate 'train' and 'test' directories
train_ds = image_dataset_from_directory(
    dataset_dir + '/train',
    labels='inferred',
    label_mode='categorical',
    color_mode='grayscale',
    batch_size=32,
    image_size=(32, 32),
    shuffle=True,
    seed=0,
)

test_ds = image_dataset_from_directory(
    dataset_dir + '/test',
    labels='inferred',
    label_mode='categorical',
    color_mode='grayscale',
    batch_size=32,
    image_size=(32, 32),
    shuffle=True,
    seed=0,
)

# Data augmentation with additional options
data_augmentation = Sequential([
    RandomFlip("horizontal_and_vertical"),
    RandomRotation(0.2),
    RandomContrast(0.2),
    RandomZoom(0.2),
])


# Load the CIFAR-10 dataset
(train_images, train_labels), (test_images, test_labels) = cifar10.load_data()
train_images, test_images = train_images / 255.0, test_images / 255.0


print(train_images[0:5])


# Load CIFAR-10 dataset
(train_images, train_labels), (test_images, test_labels) = cifar10.load_data()

# Flatten the images to 1D arrays (assuming you are using images as features)
X = train_images.reshape((train_images.shape[0], -1))
Y = train_labels

# Split the data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2, shuffle=True)

print(X.shape, X_train.shape, X_test.shape)

# Scaling the data
X_train_scaled = X_train / 255.0
X_test_scaled = X_test / 255.0
#scaling the data 
# Reshape images to (32, 32, 3)
X_train_scaled = X_train_scaled.reshape((X_train_scaled.shape[0], 32, 32, 3))


num_of_classes = 10

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(32, 32, 3)),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(num_of_classes, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['acc'])

model.fit(X_train_scaled, Y_train, validation_split=0.1, epochs=100)
from tensorflow.keras.applications import ResNet50
from tensorflow.keras import models, layers, optimizers
#build the pre trainrd model Resnet50
convolutional_base = ResNet50(weights='imagenet', include_top=False, input_shape=(256, 256, 3))
convolutional_base.summary()
num_of_classes = 10

model = models.Sequential()
model.add(layers.UpSampling2D((2,2)))
model.add(layers.UpSampling2D((2,2)))
model.add(layers.UpSampling2D((2,2)))
model.add(convolutional_base)
model.add(layers.Flatten())
model.add(layers.BatchNormalization())
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.BatchNormalization())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.BatchNormalization())
model.add(layers.Dense(num_of_classes, activation='softmax'))
model.compile(optimizer=optimizers.RMSprop(lr=2e-5), loss='sparse_categorical_crossentropy', metrics=['acc'])

history = model.fit(X_train_scaled, Y_train, validation_split=0.1, epochs=10)
loss, accuracy = model.evaluate(X_test_scaled, Y_test)
print('Test Accuracy =', accuracy)
import matplotlib.pyplot as plt

# Plot the loss value
plt.plot(history.history['loss'], label='train loss')
plt.plot(history.history['val_loss'], label='validation loss')
plt.legend()
plt.show()

# Plot the accuracy value
plt.plot(history.history['acc'], label='train accuracy')
plt.plot(history.history['val_acc'], label='validation accuracy')
plt.legend()
plt.show()
