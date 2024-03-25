"""

*** DISCLAIMER *** 

## Credits

This project utilizes code from the following sources:

https://www.elen.ucl.ac.be/Proceedings/esann/esannpdf/es2017-50.pdf
Please ensure to cite Mosca et al (2017) if any of the code following is used.

"""


import numpy as np

from keras.datasets import fashion_mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras import backend as K

# REPRODUCIBILITY
np.random.seed(404)

# LOAD DATA - FMNIST DATASET
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

X_train = X_train.reshape(60000, 28, 28, 1)
X_test = X_test.reshape(10000, 28, 28, 1)

# RANDOMLY SORT X_test/y_test
indexes = np.arange(X_test.shape[0])
for _ in range(5): indexes = np.random.permutation(indexes)  # shuffle 5 times!
X_test = X_test[indexes]
y_test = y_test[indexes]

#Implement WAME optimizer using Keras backend

from keras.optimizers import Optimizer
from keras import backend as K

if K.backend() == 'tensorflow':
    import tensorflow as tf


class WAME(Optimizer):
    
    def __init__(self, learning_rate = 0.0001, alpha=0.9, eta_plus = 1.2, eta_minus = 0.1,
                 zeta_min = 0.01, zeta_max = 100, epsilon = 1e-12, **kwargs):
        

