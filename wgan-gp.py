# completely based on https://github.com/eriklindernoren/Keras-GAN/blob/master/wgan_gp/wgan_gp.py 

# Large amount of credit goes to:
# https://github.com/keras-team/keras-contrib/blob/master/examples/improved_wgan.py
# which I've used as a reference for this implementation

from __future__ import print_function, division

from keras.datasets import mnist
from keras.layers.merge import _Merge
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, Lambda, Layer
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv1D, Conv2DTranspose
from keras.models import Sequential, Model
from keras.optimizers import RMSprop
from functools import partial
from audio_loader import load_all
from audio_tools import count_convolutions
from playsound import play_and_save_sound, save_sound

import keras.backend as K

import matplotlib.pyplot as plt

import sys
import os
import keras

import numpy as np

class Conv1DTranspose(Layer):
    def __init__(self, filters, kernel_size, strides=1, *args, **kwargs):
        self._filters = filters
        self._kernel_size = (1, kernel_size)
        self._strides = (1, strides)
        self._args, self._kwargs = args, kwargs
        super(Conv1DTranspose, self).__init__()

    def build(self, input_shape):
        #print("build", input_shape)
        self._model = Sequential()
        self._model.add(Lambda(lambda x: K.expand_dims(x,axis=1), batch_input_shape=input_shape))
        self._model.add(Conv2DTranspose(self._filters,
                                        kernel_size=self._kernel_size,
                                        strides=self._strides,
                                        *self._args, **self._kwargs))
        self._model.add(Lambda(lambda x: x[:,0]))
        self._model.summary()
        super(Conv1DTranspose, self).build(input_shape)

    def call(self, x):
        return self._model(x)

    def compute_output_shape(self, input_shape):
        return self._model.compute_output_shape(input_shape)

class RandomWeightedAverage(_Merge):
    """Provides a (random) weighted average between real and generated image samples"""
    def _merge_function(self, inputs):
        alpha = K.random_uniform((32, 1, 1))
        return (alpha * inputs[0]) + ((1 - alpha) * inputs[1])

class WGANGP():
    def __init__(self):
        os.environ["CUDA_VISIBLE_DEVICES"]="0"
        x_train = load_all("categorized", "cat",forceLoad=True,framerate=32768)
        self.X_TRAIN = x_train
        self.samples = x_train.shape[1]
        self.channels = 1
        self.kernel_size = 5
        self.audio_shape = (self.samples, self.channels)
        self.latent_dim = 100

        # Following parameter and optimizer set as recommended in paper
        self.n_critic = 5
        optimizer = RMSprop(lr=0.00005)

        # Build the generator and critic
        self.generator = self.build_generator()
        self.critic = self.build_critic()

        #-------------------------------
        # Construct Computational Graph
        #       for the Critic
        #-------------------------------

        # Freeze generator's layers while training critic
        self.generator.trainable = False

        # Image input (real sample)
        real_clip = Input(shape=self.audio_shape)

        # Noise input
        z_disc = Input(shape=(100,))
        # Generate image based of noise (fake sample)
        fake_clip = self.generator(z_disc)

        # Discriminator determines validity of the real and fake images
        fake = self.critic(fake_clip)
        valid = self.critic(real_clip)

        # Construct weighted average between real and fake images
        interpolated_clip = RandomWeightedAverage()([real_clip, fake_clip])
        # Determine validity of weighted sample
        print("Look at meeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee:") 
        print(interpolated_clip)
        validity_interpolated = self.critic(interpolated_clip)

        # Use Python partial to provide loss function with additional
        # 'averaged_samples' argument
        partial_gp_loss = partial(self.gradient_penalty_loss,
                          averaged_samples=interpolated_clip)
        partial_gp_loss.__name__ = 'gradient_penalty' # Keras requires function names

        self.critic_model = Model(inputs=[real_clip, z_disc],
                            outputs=[valid, fake, validity_interpolated])
        self.critic_model.compile(loss=[self.wasserstein_loss,
                                              self.wasserstein_loss,
                                              partial_gp_loss],
                                        optimizer=optimizer,
                                        loss_weights=[1, 1, 10])
        #-------------------------------
        # Construct Computational Graph
        #         for Generator
        #-------------------------------

        # For the generator we freeze the critic's layers
        self.critic.trainable = False
        self.generator.trainable = True

        # Sampled noise for input to generator
        z_gen = Input(shape=(100,))
        # Generate images based of noise
        img = self.generator(z_gen)
        # Discriminator determines validity
        valid = self.critic(img)
        # Defines generator model
        self.generator_model = Model(z_gen, valid)
        self.generator_model.compile(loss=self.wasserstein_loss, optimizer=optimizer)


    def gradient_penalty_loss(self, y_true, y_pred, averaged_samples):
        """
        Computes gradient penalty based on prediction and weighted real / fake samples
        """
        gradients = K.gradients(y_pred, averaged_samples)[0]
        # compute the euclidean norm by squaring ...
        gradients_sqr = K.square(gradients)
        #   ... summing over the rows ...
        gradients_sqr_sum = K.sum(gradients_sqr,
                                  axis=np.arange(1, len(gradients_sqr.shape)))
        #   ... and sqrt
        gradient_l2_norm = K.sqrt(gradients_sqr_sum)
        # compute lambda * (1 - ||grad||)^2 still for each single sample
        gradient_penalty = K.square(1 - gradient_l2_norm)
        # return the mean as loss over all the batch samples
        return K.mean(gradient_penalty)


    def wasserstein_loss(self, y_true, y_pred):
        return K.mean(y_true * y_pred)

    def build_generator(self):

        model = Sequential()
        dim = 64
        kernel_len = 25

        #convolution_layers = count_convolutions(self.audio_shape, self.kernel_size)
        convolution_layers = 3

        model.add(Dense(80 * dim, input_dim=self.latent_dim))
        model.add(Reshape((80,dim)))
        model.add(BatchNormalization())
        model.add(Activation("relu"))
        model.add(Conv1DTranspose(filters=16*dim, kernel_size=kernel_len, strides = 4, padding="same"))
        model.add(BatchNormalization())
        model.add(Activation("relu"))
        model.add(Conv1DTranspose(filters=8*dim, kernel_size=kernel_len, strides = 4, padding="same"))
        model.add(BatchNormalization())
        model.add(Activation("relu"))
        model.add(Conv1DTranspose(filters=4*dim, kernel_size=kernel_len, strides = 4, padding="same"))
        model.add(BatchNormalization())
        model.add(Activation("relu"))
        model.add(Conv1DTranspose(filters=2*dim, kernel_size=kernel_len, strides = 4, padding="same"))
        model.add(BatchNormalization())
        model.add(Activation("relu"))
        model.add(Conv1DTranspose(filters=dim, kernel_size=kernel_len, strides = 4, padding="same"))
        model.add(BatchNormalization())
        model.add(Activation("relu"))
        model.add(Conv1DTranspose(filters=1, kernel_size=kernel_len, strides = 2, padding="same"))
        model.add(BatchNormalization())
        model.add(Activation("sigmoid"))

        model.summary()

        noise = Input(shape=(self.latent_dim,))
        clip = model(noise)

        return Model(noise, clip)

    def build_critic(self):

        model = Sequential()

        convolution_layers = count_convolutions(self.audio_shape, self.kernel_size)

        input_shape = (self.audio_shape[1],1)

        model = keras.models.Sequential()

        model.add(Conv1D(16, kernel_size=self.kernel_size, activation='selu', strides=2, input_shape=self.audio_shape, padding="same"))
        for i in range(convolution_layers):
            model.add(Conv1D(32, kernel_size=self.kernel_size, activation='selu', strides=2,padding="same"))
        model.add(Flatten())
        model.add(Dropout(0.5))
        model.add(Dense(32, activation='selu'))
        model.add(Dropout(0.5))
        model.add(Dense(1))

        model.summary()

        clip = Input(shape=self.audio_shape)
        validity = model(clip)

        return Model(clip, validity)

    def train(self, epochs, batch_size, sample_interval=50):

        # Load the dataset
        X_train = self.X_TRAIN / 65536
        X_train = X_train + 0.5

        # Adversarial ground truths
        valid = -np.ones((batch_size, 1))
        fake =  np.ones((batch_size, 1))
        dummy = np.zeros((batch_size, 1)) # Dummy gt for gradient penalty
        for epoch in range(epochs):

            for _ in range(self.n_critic):

                # ---------------------
                #  Train Discriminator
                # ---------------------

                # Select a random batch of images
                idx = np.random.randint(0, X_train.shape[0], batch_size)
                imgs = X_train[idx]
                # Sample generator input
                #noise = np.zeros((self.latent_dim,1))
                noise = np.random.normal(0, 0.01, (batch_size, self.latent_dim))
                # Train the critic
                d_loss = self.critic_model.train_on_batch([imgs, noise],
                                                                [valid, fake, dummy])

            # ---------------------
            #  Train Generator
            # ---------------------

            g_loss = self.generator_model.train_on_batch(noise, valid)

            # Plot the progress
            print ("%d [D loss: %f] [G loss: %f]" % (epoch, d_loss[0], g_loss))

            # If at save interval => save generated image samples
            if epoch % sample_interval == 0:
                self.sample_clips(epoch)

    def sample_clips(self, epoch):
        r, c = 5, 5
        #noise = np.zeros(self.latent_dim)
        noise = np.random.normal(0, 0.01, (r * c, self.latent_dim))
        gen_clips = self.generator.predict(noise)

        save_sound(gen_clips, "wgan", "clap", epoch)
        #play a sound
        print("Play a sound")


if __name__ == '__main__':
    wgan = WGANGP()
    wgan.train(epochs=30000, batch_size=32, sample_interval=20)