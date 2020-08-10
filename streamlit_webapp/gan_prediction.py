import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from keras import backend as K
from keras.models import load_model
from numpy import expand_dims
from numpy import zeros
from numpy import ones
from numpy import asarray
from numpy.random import randn
from numpy.random import randint
from keras.datasets.fashion_mnist import load_data
from keras.optimizers import Adam
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Reshape
from keras.layers import Flatten
from keras.layers import Conv2D
from keras.layers import Conv2DTranspose
from keras.layers import LeakyReLU
from keras.layers import Dropout
from keras.layers import Embedding
from keras.layers import Concatenate
from keras.models import load_model
import matplotlib.pyplot as pyplot
import pickle
from tqdm import notebook

def predict(model, emotion:'int'):
	x_input = randn(100 * 1)
	z_input = x_input.reshape(1, 100)
	X = model.predict([z_input, asarray([emotion])])
	X = (X + 1) / 2.0
	pyplot.imsave("generated_img.jpg", X[0,:,:,0], cmap="gray")

if __name__ == '__main__':
	model = load_model('./cgan_weight.h5')
	predict(model, 4)