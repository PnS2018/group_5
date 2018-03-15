"""Convolutional Neural Network for Fashion MNIST Classification.

Team #name
"""
from __future__ import print_function, absolute_import

import numpy as np
import matplotlib.pyplot as plt

from keras.layers import Input, Dense, Conv2D, MaxPooling2D, Flatten, Activation
from keras.models import Model
from keras.utils import to_categorical

from pnslib import utils
from pnslib import ml

# Load all the ten classes from Fashion MNIST
# complete label description is at
# https://github.com/zalandoresearch/fashion-mnist#labels
(train_x, train_y, test_x, test_y) = utils.fashion_mnist_load(
    data_type="full", flatten=False)

num_classes = 10

print ("[MESSAGE] Dataset is loaded.")

# preprocessing for training and testing images
train_x = train_x.astype("float32")/255.  # rescale image
mean_train_x = np.mean(train_x, axis=0)  # compute the mean across pixels
train_x -= mean_train_x  # remove the mean pixel value from image
test_x = test_x.astype("float32")/255.
test_x -= mean_train_x

print ("[MESSAGE] Dataset is preprocessed.")

# converting the input class labels to categorical labels for training
train_Y = to_categorical(train_y, num_classes=num_classes)
test_Y = to_categorical(test_y, num_classes=num_classes)

print("[MESSAGE] Converted labels to categorical labels.")

# define a model
# >>>>> PUT YOUR CODE HERE <<<<<
x = Input((train_x.shape[1],train_x.shape[2], train_x.shape[3] ), name="input_layer") #instead of train_X.shape[1]$
dens1= Conv2D(filters=10,          # this layer has 20 filters, number of filters equals number of kernels
           kernel_size=(7, 7),  # the filter size is 7X7
           strides=(2, 2),      # horizontal stride is 2, vertical stride is 2
           padding="same")(x)   # pad 0s so that the output has the same shape as input
dens1= Activation("relu", name="activation_dens1")(dens1)
dens2= Conv2D(filters=10,          # this layer has 25 filters, number of filters equals number of kernels
           kernel_size=(5, 5),  # the filter size is 5x5
           strides=(2, 2),      # horizontal stride is 2, vertical stride is 2
           padding="same")(dens1)   # pad 0s so that the output has the same shape as input
dens2 = Activation("relu", name="activation_dens2")(dens2)
y = Flatten()(dens2)
y = Dense(200, name="space_with_200_units")(y)
y = Activation("relu", name="activation_y_200_units")(y)
y = Dense(10, name="output_layer")(y)
y = Activation("softmax", name="activation_y")(y)
model = Model(x, y)




print("[MESSAGE] Model is defined.")

# print model summary
model.summary()

# compile the model aganist the categorical cross entropy loss
# and use SGD optimizer, you can try to use different
# optimizers if you want
# see https://keras.io/losses/
# >>>>> PUT YOUR CODE HERE <<<<<

model.compile(loss="categorical_crossentropy", optimizer ="sgd", metrics=["accuracy"])


print ("[MESSAGE] Model is compiled.")

# train the model with fit function
# See https://keras.io/models/model/ for usage
# >>>>> PUT YOUR CODE HERE <<<<<
num_epochs = 2
batch_size = 128
model.fit(x=train_x, y=train_Y,
    batch_size=batch_size, epochs=num_epochs,
    validation_data=(test_x, test_Y))




print("[MESSAGE] Model is trained.")

# save the trained model
model.save("conv-net-fashion-mnist-trained.hdf5")

print("[MESSAGE] Model is saved.")

# visualize the ground truth and prediction
# take first 10 examples in the testing dataset
test_x_vis = test_x[:10]  # fetch first 10 samples
ground_truths = test_y[:10]  # fetch first 10 ground truth prediction
# predict with the model
preds = np.argmax(model.predict(test_x_vis), axis=1).astype(np.int)

labels = ["Tshirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal",
          "Shirt", "Sneaker", "Bag", "Ankle Boot"]

plt.figure()
for i in xrange(2):
    for j in xrange(5):
        plt.subplot(2, 5, i*5+j+1)
        plt.imshow(test_x[i*5+j, :, :, 0], cmap="gray")
        plt.title("Ground Truth: %s, \n Prediction %s" %
                  (labels[ground_truths[i*5+j]],
                   labels[preds[i*5+j]]))
plt.show()
