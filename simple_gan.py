# based on https://github.com/eriklindernoren/Keras-GAN/blob/master/gan/gan.py
from __future__ import print_function, division

import keras
from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv1D
from keras.models import Sequential, Model
from keras.optimizers import Adam
from audio_loader import load_audio
from audio_tools import count_convolutions
from playsound import play_and_save_sound

import sys

import numpy as np

class GAN():
    def __init__(self):
        self.num_classes = 10
        (x_train, y_train), (x_test, y_test) = load_audio("speech_commands", self.num_classes)
        self.X_TRAIN = x_train
        self.samples = x_train.shape[1]
        self.channels = 1
        self.kernel_size = 5
        self.audio_shape = (self.samples, self.channels)
        self.latent_dim = 100

        optimizer = Adam(0.0002, 0.5)

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy',
            optimizer=optimizer,
            metrics=['accuracy'])

        # Build the generator
        self.generator = self.build_generator()

        # The generator takes noise as input and generates audio
        z = Input(shape=(self.latent_dim,))
        audio = self.generator(z)

        # For the combined model we will only train the generator
        self.discriminator.trainable = True

        # The discriminator takes generated audio as input and determines validity
        validity = self.discriminator(audio)

        # The combined model  (stacked generator and discriminator)
        # Trains the generator to fool the discriminator
        self.combined = Model(z, validity)
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)


    def build_generator(self):

        model = Sequential()

        model.add(Dense(256, input_dim=self.latent_dim))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(1024))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(self.samples, activation='tanh'))
        model.add(Reshape(self.audio_shape))

        model.summary()

        noise = Input(shape=(self.latent_dim,))
        clip = model(noise)

        return Model(noise, clip)

    def build_discriminator(self):

        convolution_layers = count_convolutions(self.audio_shape, self.kernel_size)

        model = keras.models.Sequential()
        model.add(Conv1D(16, kernel_size=self.kernel_size, activation='selu', strides=2, input_shape=self.audio_shape))
        for i in range(convolution_layers):
            model.add(Conv1D(32, kernel_size=self.kernel_size, activation='selu', strides=2))
        model.add(Flatten())
        model.add(Dropout(0.5))
        model.add(Dense(32, activation='selu'))
        model.add(Dropout(0.5))

        model.add(Dense(1, activation='sigmoid'))
        model.summary()

        img = Input(shape=self.audio_shape)
        validity = model(img)

        return Model(img, validity)

    def train(self, epochs, batch_size=128, sample_interval=50):

        # Adversarial ground truths
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        X_train = self.X_TRAIN

        for epoch in range(epochs):

            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Select a random batch of images
            idx = np.random.randint(0, X_train.shape[0], batch_size)
            clips = X_train[idx]

            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))

            # Generate a batch of new images
            gen_clips = self.generator.predict(noise)

            # Train the discriminator
            d_loss_real = self.discriminator.train_on_batch(clips, valid)
            d_loss_fake = self.discriminator.train_on_batch(gen_clips, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ---------------------
            #  Train Generator
            # ---------------------

            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))

            # Train the generator (to have the discriminator label samples as valid)
            g_loss = self.combined.train_on_batch(noise, valid)

            # Plot the progress
            print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))

            # If at save interval => save generated image samples
            if epoch % sample_interval == 0:
                self.sample_clips(epoch)

    def sample_clips(self, epoch):
        r, c = 5, 5
        noise = np.random.normal(0, 1, (r * c, self.latent_dim))
        gen_clips = self.generator.predict(noise)

        play_and_save_sound(gen_clips, "generated")
        #play a sound
        print("Play a sound")


if __name__ == '__main__':
    gan = GAN()
    gan.train(epochs=30000, batch_size=32, sample_interval=200)