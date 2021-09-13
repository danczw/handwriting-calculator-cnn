# imports
import numpy as np
from PIL import Image, ImageDraw
from tensorflow import keras
from tensorflow.keras import layers
import tkinter as tk

#------------------------------------------------------------------------------#
#                               Data Preparation                               #
#------------------------------------------------------------------------------#

# set image sizes
img_w, img_h = 28, 28

# load, split and data
(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()

# reshape data to match keras format
X_train = X_train.reshape(-1, img_w, img_h, 1)
X_test = X_test.reshape(-1, img_w, img_h, 1)

# normalize data  # TODO: does keras.utils.normalize do the trick?
X_train = X_train.astype('float32') / 255
X_test = X_test.astype('float32') / 255

# translate to categorical labels
y_train_cat = keras.utils.to_categorical(y_train)
y_test_cat = keras.utils.to_categorical(y_test)

#------------------------------------------------------------------------------#
#                         Convolutional Neural Network                         #
#------------------------------------------------------------------------------#

# set model parameter
img_size = (img_w, img_h, 1)
pool_size = (2, 2)
kernel_tupel = (3, 3)
dropout_ratio = 0.25

# define mode
model = keras.models.Sequential()

model.add(layers.Conv2D(32, kernel_tupel, activation='relu', input_shape=img_size))
model.add(layers.MaxPooling2D(pool_size))
model.add(layers.Dropout(dropout_ratio))

model.add(layers.Conv2D(64, kernel_tupel, activation='relu', input_shape=img_size))
model.add(layers.MaxPooling2D(pool_size))
model.add(layers.Dropout(dropout_ratio))

model.add(layers.Conv2D(128, kernel_tupel, activation='relu', input_shape=img_size))
model.add(layers.MaxPooling2D(pool_size))
model.add(layers.Dropout(dropout_ratio))

model.add(layers.Flatten())

model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

print(model.summary())

# fit model
epochs = 5
batch = 32
val_split = 0.3

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
history_test = model.fit(X_train, y_train_cat,
    epochs=epochs,
    verbose=1, 
    validation_split=val_split,
    batch_size=batch
)

# show test accuracy
score = model.evaluate(X_test, y_test_cat, verbose=0)
print('Test accuracy:', score[1])
