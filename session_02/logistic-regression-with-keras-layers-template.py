"""Logistic Regression for Fashion MNIST Binary Classification.

Team #name
"""
from __future__ import print_function, absolute_import

import numpy as np
import matplotlib.pyplot as plt

from keras.layers import Input, Dense, Activation
from keras.models import Model

from pnslib import utils
from pnslib import ml

# Load T-shirt/top and Trouser classes from Fashion MNIST
# complete label description is at
# https://github.com/zalandoresearch/fashion-mnist#labels
# >>>>> Try to change the class list for other classes <<<<<
(train_x, train_y, test_x, test_y) = utils.binary_fashion_mnist_load(
    class_list=[8,9],
    flatten=True)

print ("[MESSAGE] Dataset is loaded.")


# preprocessing for training and testing images
train_x = train_x.astype("float32")/255.  # rescale image
mean_train_x = np.mean(train_x, axis=0)  # compute the mean across pixels
train_x -= mean_train_x  # remove the mean pixel value from image
test_x = test_x.astype("float32")/255.
test_x -= mean_train_x

print ("[MESSAGE] Dataset is preporcessed.")

# Use PCA to reduce the dimension of the dataset,
# so that the training will be less expensive
# perform PCA on training dataset
train_X, R, n_retained = ml.pca(train_x)

# perform PCA on testing dataset
test_X = ml.pca_fit(test_x, R, n_retained)

print ("[MESSAGE] PCA is complete")

# define a model
# >>>>> PUT YOUR CODE HERE <<<<<
x = Input((train_X.shape[1],), name="input_layer") #instead of train_X.shape%1] write 340
y = Dense(1, name="linear_layer")(x)  # suppose there are 5 classes
y = Activation("sigmoid")(y)
model = Model(x, y)




# print model summary
model.summary()

# compile the model aganist the binary cross entropy loss
# and use SGD optimizer, you can try to use different
# optimizers if you want
# see https://keras.io/losses/
# >>>>> PUT YOUR CODE HERE <<<<<
model.compile(loss="binary_crossentropy",
                    optimizer="sgd",
                    metrics=["accuracy"])





print ("[MESSAGE] Model is compiled")

# train the model with fit function
# See https://keras.io/models/model/ for usage
# >>>>> PUT YOUR CODE HERE <<<<<
num_epochs = 50
batch_size = 128
model.fit(x=train_X, y=train_y,
    batch_size=batch_size, epochs=num_epochs,
    validation_data=(test_X, test_y))





# save the trained model
model.save("logistic-regression-fashion-mnist-trained.hdf5")

# visualize the ground truth and prediction
# take first 10 examples in the testing dataset
test_X_vis = test_X[:10]  # fetch first 10 samples
ground_truths = test_y[:10]  # fetch first 10 ground truth prediction
# predict with the model
preds = (model.predict(test_X_vis) > 0.5)[:, 0].astype(np.int)

labels = ["bag", "Ankle boot"]

plt.figure()
for i in xrange(2):
    for j in xrange(5):
        plt.subplot(2, 5, i*5+j+1)
        plt.imshow(test_x[i*5+j].reshape(28, 28), cmap="gray")
        plt.title("Ground Truth: %s, \n Prediction %s" %
                  (labels[ground_truths[i*5+j]],
                   labels[preds[i*5+j]]))
plt.show()
