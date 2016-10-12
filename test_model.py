'''Train a simple convnet on the MNIST dataset.
Run on GPU: THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python mnist_cnn.py
Get to 99.25% test accuracy after 12 epochs (there is still a lot of margin for parameter tuning).
16 seconds per epoch on a GRID K520 GPU.
'''

from __future__ import print_function
import numpy as np
import gzip
from six.moves import cPickle
import sys

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.models import model_from_json

def load_data1(path="mnist.pkl.gz"):
    #path = get_file(path, origin="https://s3.amazonaws.com/img-datasets/mnist.pkl.gz")

    if path.endswith(".gz"):
        f = gzip.open(path, 'rb')
    else:
        f = open(path, 'rb')

    if sys.version_info < (3,):
        data = cPickle.load(f)
    else:
        data = cPickle.load(f, encoding="bytes")

    f.close()
    return data  # (X_train, y_train), (X_test, y_test)

batch_size = 20
nb_classes = 3
nb_epoch = 50

# input image dimensions
#img_rows, img_cols = 30, 33
img_rows, img_cols = 30, 33
# number of convolutional filters to use
nb_filters = 64
# size of pooling area for max pooling
nb_pool = 7
# convolution kernel size
nb_conv = 5

# the data, shuffled and split between train and test sets
(X_train, y_train), (X_test, y_test) = load_data1(path='dataStroke2.pkl.gz')
#(X_train, y_train), (X_test, y_test) = load_data1(path='dataNew2.pkl.gz')
X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
#X_train /= 255
#X_test /= 255
print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

#model = model_from_json(open('modelClass_1.json').read())
mypath = '/Users/Mint/Documents/2_2558/Deep neural network/Code/my_model/'
model = model_from_json(open(mypath+'my_model_architecture_30.json').read())
model.load_weights(mypath+'my_model_weights_30.h5')
model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=["accuracy"])
y_pred = model.predict(X_test)

score = np.argmax(Y_test, axis=1)-np.argmax(y_pred, axis=1)




#print(np.argmax(y_pred, axis=1))

#print(score)
score = score.tolist()
print(score.count(-1))
print(np.argmax(Y_test, axis=1).tolist().count(0))
print(1-score.count(-1)/np.argmax(Y_test, axis=1).tolist().count(0))
print("\n")
print(score.count(1))
print(np.argmax(Y_test, axis=1).tolist().count(1))
print(1-score.count(1)/np.argmax(Y_test, axis=1).tolist().count(1))
print(score.count(0)/X_test.shape[0])
