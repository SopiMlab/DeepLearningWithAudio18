import keras
from pydub import AudioSegment
from keras.models import Sequential, Model
from keras.layers.advanced_activations import LeakyReLU
from keras.layers import Input, Dense, Reshape, Flatten, BatchNormalization, Activation, Layer, Lambda
from keras.layers.convolutional import UpSampling2D, Conv1D, Conv2DTranspose
import numpy as np
from playsound import play_sound, play_and_save_sound
from audio_loader import load_all

import keras.backend as K


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

latent_dim = 100
sound = load_all("categorized", "cat",forceLoad=True)

#play_and_save_sound(sound, "mimic", "original", upscale=False)

sound = sound / 65536
sound = sound + 0.5
target = np.array(sound[0])
target = target.reshape(1, target.shape[0], target.shape[1])
print(target.shape)

input_shape = (1,target.shape[0])
print(input_shape)

samples = len(sound[0])
print(samples)

#play_and_save_sound(sound, "mimic", "normalized", upscale=True)

model = Sequential()

dim = 64
kernel_len = 5

#convolution_layers = count_convolutions(self.audio_shape, self.kernel_size)
convolution_layers = 3

model.add(Dense(1280 * 16, input_dim=latent_dim))
model.add(Reshape((1280,16)))
model.add(BatchNormalization())
model.add(Activation("selu"))
for i in range(convolution_layers):
    model.add(Conv1DTranspose(filters=32, kernel_size=kernel_len, strides = 4, padding="same"))
    model.add(BatchNormalization())
    model.add(Activation("selu"))
model.add(Conv1DTranspose(filters=1, kernel_size=kernel_len, strides = 2, padding="same"))
model.add(BatchNormalization())
model.add(Activation("tanh"))

# model.add(Dense(256, input_dim=latent_dim))
# model.add(LeakyReLU(alpha=0.2))
# model.add(BatchNormalization(momentum=0.8))
# model.add(Dense(512))
# model.add(LeakyReLU(alpha=0.2))
# model.add(BatchNormalization(momentum=0.8))
# model.add(Dense(1024))
# model.add(LeakyReLU(alpha=0.2))
# model.add(BatchNormalization(momentum=0.8))
# model.add(Dense(samples, activation='tanh'))
# model.add(Reshape([samples,1]))

model.summary()

noise = Input(shape=(latent_dim,))
clip = model(noise)

generator = Model(noise, clip)

noise = np.random.normal(0, 1, (1, latent_dim))
gen_clip = generator.predict(noise, 1)

gen_clip = gen_clip + 0.5

play_and_save_sound(gen_clip, "mimic", "startingnoise")

model.compile(loss=keras.losses.mean_squared_error,
              optimizer=keras.optimizers.Adam())

for i in range(100):
    #gen_clip = generator.predict(noise, 1
    print("training: " + str(model.train_on_batch(noise,target)))
    if i % 5 == 0:
        gen_clip = generator.predict(noise, 1)
        play_and_save_sound(gen_clip, "mimic", "generated", epoch=i)