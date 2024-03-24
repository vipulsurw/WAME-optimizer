"""

*** DISCLAIMER *** 

## Credits

This project utilizes code from the following sources:


https://www.elen.ucl.ac.be/Proceedings/esann/esannpdf/es2017-50.pdf
  -Please ensure to cite Mosca et al (2017) if any of the code following is used. Available from: 

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

