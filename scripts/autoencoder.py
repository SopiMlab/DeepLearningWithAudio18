import sys
sys.path.insert(0, 'tools')

from keras.layers import Input, Dense, Conv1D, Layer, Lambda, Conv2DTranspose
from keras.models import Model, Sequential
import os
from audio_loader import load_audio
import numpy as n
from audio_tools import count_convolutions
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

# this is the size of our encoded representations
encoding_dim = 32  # 32 floats -> compression of factor 24.5, assuming the input is 784 floats

os.environ["CUDA_VISIBLE_DEVICES"]="0"
(x_train, _), (x_test, _) = load_audio("nsynth", 10, framerate=16384, forceLoad=True)
samples = x_train.shape[1]
print(samples)
channels = 1
input_shape = (x_train.shape[1],1)
kernel_size = 5
folder_name = "nsynthnormalized"
# this is our input placeholder
input_clip = Input(shape=(samples,1,))
# "encoded" is the encoded representation of the input

convolution_layers = count_convolutions(input_shape, kernel_size)

encoded = Conv1D(16, kernel_size=kernel_size, activation='selu', strides=2, input_shape=input_shape, padding="same")(input_clip)
for i in range(convolution_layers):
    encoded = Conv1D(16, kernel_size=kernel_size, activation='selu', strides=2,padding="same")(encoded)

decoded = Conv1DTranspose(16,kernel_size,strides=2, padding='same', activation='selu')(encoded)
for i in range(convolution_layers - 1):
    decoded = Conv1DTranspose(16, kernel_size=kernel_size, activation='selu', strides=2,padding="same")(decoded)
decoded = Conv1DTranspose(1, kernel_size=kernel_size, activation='selu', strides=2,padding="same")(decoded)
# encoded = Dense(4096, activation='relu')(input_clip)
# encoded = Dense(1024, activation='relu')(encoded)
# encoded = Dense(256, activation='relu')(encoded)
# encoded = Dense(64, activation='relu')(encoded)
# encoded = Dense(encoding_dim, activation='relu')(encoded)


# decoded = Dense(64, activation='relu')(encoded)
# decoded = Dense(256, activation='relu')(decoded)
# decoded = Dense(1024, activation='relu')(decoded)
# decoded = Dense(4096, activation='relu')(decoded)
# decoded = Dense(samples, activation='sigmoid')(decoded)



# this model maps an input to its reconstruction
autoencoder = Model(input_clip, decoded)

autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

autoencoder.summary()

x_train = x_train.astype('float32') / 65536
x_train = x_train + 0.5
x_test = x_test.astype('float32') / 65536
x_test = x_test + 0.5

print (x_train.shape)

autoencoder.fit(x_train, x_train,
                epochs=50,
                batch_size=32,
                shuffle=True,
                validation_data=(x_test, x_test))

from playsound import save_sound

decoded_imgs = autoencoder.predict(x_test)

n = 10  # how many digits we will display
for i in range(n):
    save_sound(x_test,folder_name,"original",upscale=True,index=i, epoch=i)
    save_sound(decoded_imgs,folder_name,"decoded",upscale=True,index=i, epoch=i)