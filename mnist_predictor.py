import tensorflow as tf
import numpy as np 
import keras



from keras.datasets import mnist 
(x_train, y_train), (x_test, y_test) = mnist.load_data()


from keras.layers import Dense, Conv2D, Dropout, Flatten, GaussianNoise, GaussianDropout
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential 
from keras.callbacks import TensorBoard
from keras.optimizers import Nadam



#getting data in the right shape

#input data
s1 = x_test.shape
x_test = np.reshape(x_test, [s1[0], s1[1], s1[2], 1])
s2 = x_train.shape
x_train = np.reshape(x_train, [s2[0], s2[1], s2[2], 1])

#output data
y_training = []
for t in y_train:
	inter = np.zeros(10)
	inter[t] = 1
	y_training.append(inter)

y_testing = []
for t in y_test:
	inter = np.zeros(10)
	inter[t] = 1
	y_testing.append(inter)


#only select some of the data
x_train = x_train[:500]
y_training = y_training[:500]
x_test = x_test[:200]
y_testing = y_testing[:200]

#normalize
def norm(data):
	data = (data - np.min(data)) / (np.max(data) - np.min(data))
	new_data = np.array([im - np.mean(im) for im in data])
	return new_data

x_train = norm(x_train)
x_test = norm(x_test)


#create the NN
model = Sequential()

model.add(GaussianNoise(.05))

model.add(Conv2D(32, (3, 3), input_shape = (28, 28, 1), padding = 'valid'))
model.add(LeakyReLU(alpha = .3))
#model.add(Dropout(.2))
model.add(GaussianNoise(.1))

model.add(Conv2D(32, (3, 3), padding = 'valid'))
model.add(LeakyReLU(alpha = .3))
#model.add(Dropout(.2))
model.add(GaussianDropout(.2))

model.add(Flatten())

model.add(Dense(128))
model.add(LeakyReLU(alpha = .3))
#model.add(Dropout(.5))
model.add(GaussianNoise(.3))

model.add(Dense(32))
model.add(LeakyReLU(alpha = .3))
model.add(Dropout(.4))

model.add(Dense(10, activation = 'sigmoid'))

model.compile(optimizer = Nadam(), loss = 'categorical_crossentropy', metrics = ['accuracy'])

from time import time
TensorBoardLog = TensorBoard(log_dir = '/Users/loganjaeger/Desktop/mnist_trials/{}'.format(time()))
#fit the NN
model.fit(x = x_train,
 y = np.array(y_training),
 batch_size = 256,
 epochs = 30,
 validation_data = (x_test, np.array(y_testing)),
 verbose = 2,
 callbacks = [TensorBoardLog]
 )




