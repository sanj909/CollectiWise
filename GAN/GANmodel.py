import tensorflow as tf
import tensorflow.keras.layers as layers
import keras as og_keras
from keras_self_attention import SeqSelfAttention
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

#BTC.csv
data = pd.read_csv('/Users/Sanjit/Google Drive/CollectiWise/Data/BTC.csv').drop(columns=['open', 'high', 'low', 'volume'])

#Data preprocessing
#Download the files models.py, dataUtils.py and waveletDenoising.py from LSTM branch and run this script in the same folder as those files. 
from models import *
from dataUtils import *
from waveletDenoising import normalise

close_data = normalise(data.close.to_numpy(), 0, 1)

unroll_length = 10
X_train, X_test, y_train, y_test = train_test_split_lstm(close_data, 5, int(close_data.shape[0] * 0.1))
X_train = np.expand_dims(unroll(X_train, unroll_length), axis = 2)
y_train = np.expand_dims(unroll(y_train, unroll_length), axis = 2)
X_test = np.expand_dims(unroll(X_test, unroll_length), axis = 2)
y_test = np.expand_dims(unroll(y_test, unroll_length), axis = 2)

'''
    https://machinelearningmastery.com/how-to-develop-a-generative-adversarial-network-for-a-1-dimensional-function-from-scratch-in-keras/
    In the example above, generator guesses pairs (x, y). Discriminator classifies them as real or fake. 
    The generator learns to guess pairs (x, x^2). The discriminator learns to classify only pairs (x, x^2) as real. 

    We want our generator to guess a time series X, and the disciminator to classify it as real or fake. 
    Generator: MLP. Returns next unroll_length prices from a randomly generated vector of size latent_dim.
    Disciminator: MLP with final sigmoid layer

    'Training the discriminator model is straightforward. The goal is to train a generator model, not a discriminator model, 
    and that is where the complexity of GANs truly lies.'

	'When the discriminator is good at detecting fake samples, the generator is updated more, and when the discriminator model 
	is relatively poor or confused when detecting fake samples, the generator model is updated less.'

	'This is because the latent space has no meaning until the generator model starts assigning meaning to points in the space as it learns. 
	After training, points in the latent space will correspond to points in the output space, e.g. in the space of generated samples.'

	In the example, their samples were (x, y)  pairs. For us, the sample will be an unroll_length dimensional vector.
'''

#a simple discriminator model
def define_discriminator(n_inputs=unroll_length):
	model = tf.keras.models.Sequential()
	model.add(layers.Dense(25, activation='relu', kernel_initializer='he_uniform', input_dim=n_inputs))
	model.add(layers.Dense(1, activation='sigmoid'))
	#Compile model
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model

def define_generator(latent_dim, n_outputs=unroll_length):
	model = tf.keras.models.Sequential()
	model.add(layers.Dense(15, activation='relu', kernel_initializer='he_uniform', input_dim=latent_dim))
	model.add(layers.Dense(n_outputs, activation='linear'))
	return model

def define_gan(generator, discriminator):
	#Include the line below if the discriminator has been pre-trained. This kinda defeats the point of a GAN though
	#discriminator.trainable=False 
	model = tf.keras.models.Sequential()
	model.add(generator)
	model.add(discriminator)
	#Compile model
	model.compile(loss='binary_crossentropy', optimizer='adam')
	return model

def generate_real_samples(n_epoch, n): #gets n rows from X_train, which has 834 rows total
	#Line below slices first n rows when n_epoch = 1, slices next n rows when n_epoch = 2, and so on
	X1 = X_train[((n_epoch-1)*n):n_epoch*n] 
	X = X1.reshape(n, unroll_length)
	y = np.ones((n, 1))
	return X, y #1 is the class label, 1 means real

def generate_test_samples(n): #gets n rows from X_test, which has 136 rows total
	#####
	#####
	X1 = X_test[:n] #This simply gets the first n rows of X_test. Needs fixing so that we use the whole test set. 
	#####
	#####
	X = X1.reshape(n, unroll_length)
	y = np.ones((n, 1))
	return X, y #1 is the class label, 1 means real

#We must have some vector to input into the first layer of the generator. 
#This function creates that input.
def generate_latent_points(latent_dim, n):
	x_input = np.random.randn(latent_dim * n)
	x_input = x_input.reshape(n, latent_dim)
	return x_input

def generate_fake_samples(generator, latent_dim, n): #gets generator prediction
	x_input = np.random.randn(latent_dim * n)
	x_input = x_input.reshape(n, latent_dim)
	X = generator.predict(x_input)
	y = np.zeros((n, 1))
	return X, y #0 is the class label, 0 means fake

def summarise_performance(epoch, generator, disciminator, latent_dim, n=34): 
	#n = size of test set / int(n_epochs/n_eval). Here, it's 136/(4) = 34
	#My idea is that since int(n_epochs/n_eval) = 4, summarise performance will be called 4 times in train(), so each time
	#it's called, we use a quarter of the rows of X_test. Maybe each time it's called, we can use the whole of X_test???
	x_real, y_real = generate_test_samples(n) #On each successive call of this function, we want to get the next 34 rows. Right now, we just get the first 34 rows
	_, acc_real = discriminator.evaluate(x_real, y_real, verbose=0)
	x_fake, y_fake = generate_fake_samples(generator, latent_dim, n)
	_, acc_fake = discriminator.evaluate(x_fake, y_fake, verbose=0)
	print(epoch, acc_real, acc_fake)

def train(generator, disciminator, gan, latent_dim, n_epochs=417, n_batch=4, n_eval=100):
	half_batch = int(n_batch/2)
	#834 rows in X_train for BTC.csv. 
	#n_epochs must be less or equal to 834/half_batch, ideally as close as possible to this limit so that we use all the training data. 
	#In each epoch, dicriminator is trained on half_batch real samples and half_batch fake samples. We want to train at least once on each real sample in X_train.
	for i in range(1, n_epochs+1):
		x_real, y_real = generate_real_samples(i, half_batch)
		x_fake, y_fake = generate_fake_samples(generator, latent_dim, half_batch)

		disciminator.train_on_batch(x_real, y_real)
		disciminator.train_on_batch(x_fake, y_fake)

		x_gan = generate_latent_points(latent_dim, n_batch)
		y_gan = np.ones((n_batch, 1))

		gan.train_on_batch(x_gan, y_gan)
		#Line below updates discriminator weights, so both models are trained simulataneously.
		#This is a deviation from the example, where they train the discriminator first. 
		#This throws a shit ton of error messages
		#gan = define_gan(generator, discriminator) 

		#evaluate the model every n_eval epochs on the test set
		if (i+1)%n_eval == 0:
			summarise_performance(i, generator, disciminator, latent_dim)

latent_dim = 20 
discriminator = define_discriminator()
generator = define_generator(latent_dim)
gan = define_gan(generator, discriminator)
#train(generator, discriminator, gan, latent_dim) 

'''
	When we include line 118, partial output is
	299 0.0 0.9411764740943909
	399 0.23529411852359772 0.970588207244873

	When we comment out line 62, the output is 
	99 0.0 0.6470588445663452
	199 0.0 0.20588235557079315
	299 0.3235294222831726 0.0882352963089943
	399 0.5 0.0882352963089943

	Ideally, what we want to see as output is something like
	index 0.5 0.5 
	index 0.49 0.51
	index 0.51 0.49
	i.e. the generator is so good at creating fakes that the discriminator must guess at random. 

	Next step: Try this with A LOT more data. 
	If you want to save the model weights, see trainPlayground.py in LSTM branch, or just see the guide on tensorflow.org.
'''

