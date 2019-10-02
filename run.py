# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
from tensorflow import keras


""" Additional Helper Libraries """

import numpy as np
import matplotlib.pyplot as plt


""" Import the Dataset"""

fashion_mnist = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()



""" Preprocess the data """

plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(False)
plt.show()

""" 
Since the scaling is currently set to values from 0 to 250
we want our neural network to only contain values between 0 and 
1, to proceed we will divide all values by 255
 """
 
train_images = train_images / 255
test_images = test_images / 255
 
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
plt.figure(figsize = (10,10))
for i in range(25):
    plt.subplot(5, 5, i+ 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap= plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.show

"""
Build the neural network and its layers 
"""

model = keras.Sequential([keras.layers.Flatten(input_shape =(28,28)), keras.layers.Dense(128, activation = 'relu'), keras.layers.Dense(10, activation = 'softmax')])

model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])

model.fit(train_images, train_labels, epochs = 10)
