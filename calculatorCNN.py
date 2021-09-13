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
