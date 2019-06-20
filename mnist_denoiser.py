import tensorflow as tf
import numpy as np 
import keras



from keras.datasets import mnist 
(x_train, y_train), (x_test, y_test) = mnist.load_data()


from keras.layers import Dense, Conv2D, Dropout, Flatten, GaussianNoise, GaussianDropout, MaxPooling2D, UpSampling2D, BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.activations import relu
from keras.models import Sequential 
from keras.callbacks import TensorBoard, EarlyStopping
from keras.optimizers import Nadam, Adadelta, Adam



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
x_train = x_train[:700]
y_training = y_training[:700]
x_test = x_test[:250]
y_testing = y_testing[:250]

#normalize
def norm(data):
	data = (data - np.min(data)) / (np.max(data) - np.min(data))
	new_data = np.array([im - np.mean(im) for im in data])
	return new_data

x_train = norm(x_train)
x_test = norm(x_test)

def add_noise(data):
	shape = data.shape
	gauss = np.random.normal(0, .4, shape)
	return data + gauss


noise_test = add_noise(x_test)
noise_train = add_noise(x_train)

alp = .01
batch_size = 512
epochs = 30
first_conv_layers = 64
second_conv_layers = 32
opt = Adam()

#create the NN
model = Sequential()
#encoder
#model.add(GaussianNoise(.4, input_shape = (28, 28, 1)))
model.add(Conv2D(first_conv_layers, (3, 3), input_shape = (28, 28, 1), padding = 'same'))
model.add(BatchNormalization())
model.add(LeakyReLU(alpha = alp))
model.add(MaxPooling2D())

#model.add(Dropout(.3))

model.add(Conv2D(second_conv_layers, (3, 3), padding = 'same'))
model.add(BatchNormalization())
model.add(LeakyReLU(alpha = alp))
#model.add(MaxPooling2D())

#model.add(Dropout(.1))

#model.add(Conv2D(8, (3, 3), padding = 'same', activation = 'relu'))

#encoded data
model.add(MaxPooling2D(padding = 'same'))

#decoder

# model.add(Conv2D(8, (3, 3), padding = 'same', activation = 'relu'))
# model.add(UpSampling2D())

#model.add(Dropout(.1))

model.add(Conv2D(second_conv_layers, (3, 3), padding = 'same'))
model.add(BatchNormalization())
model.add(LeakyReLU(alpha = alp))
model.add(UpSampling2D())

#model.add(Dropout(.3))

model.add(Conv2D(first_conv_layers, (3, 3), padding = 'same'))
model.add(BatchNormalization())
model.add(LeakyReLU(alpha = alp))
model.add(UpSampling2D())

model.add(Conv2D(1, (3, 3), padding = 'same', activation = 'sigmoid'))

model.compile(optimizer = opt, loss = 'binary_crossentropy')

from time import time
#Checkpoint = EarlyStopping(monitor = 'val_loss', patience = 3)
TensorBoardLog = TensorBoard(log_dir = '/Users/loganjaeger/Desktop/mnist_trials/AE/2/withcheckpoint/no_xtralayers/alpha_{}/{}'.format(alp, time()))
#fit the NN
model.fit(x = noise_train,
 y = x_train,
 batch_size = batch_size,
 epochs = epochs,
 validation_data = (noise_test, x_test),
 verbose = 2,
 callbacks = [TensorBoardLog]
 )


import matplotlib.pyplot as plt 

#plot original


p = model.predict(noise_test[:10])

for i in range(10):

	plt.subplot(2, 10, i + 1)
	plt.imshow(noise_test[i].reshape(28, 28))
	plt.gray()
	plt.axis('off')
	#plt.title('original {}'.format(i))

	#plot
	plt.subplot(2, 10, i + 11)
	plt.imshow(p[i].reshape(28, 28))
	plt.gray()
	plt.axis('off')
	#plt.title('reconstructed {}'.format(i))

plt.title('alpha: {} \n batch_size: {} \n epochs: {} \n first_conv: {} \n second_conv: {} \n optimizer: {}'.format(alp, batch_size, epochs, first_conv_layers, second_conv_layers, str(opt)))
plt.show()

