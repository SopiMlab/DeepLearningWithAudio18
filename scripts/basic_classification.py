import sys
sys.path.insert(0, 'tools')

from audio_tools import count_convolutions
from audio_loader import load_audio
import keras
from keras.layers import Dense, Dropout, Flatten,LeakyReLU
from keras.layers import Conv1D 
from playsound import save_sound
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

num_classes = 2
model_savepath = "saved_model_3sec2"

(x_train, y_train), (x_test, y_test) = load_audio("beatles3sec2", num_classes, forceLoad=False, framerate=16384,amount_limit=2400)

print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)

batch_size = 128
epochs = 40
kernel_size = 5

#save_sound(x_train, "classification","xtrain",upscale=False)
#save_sound(x_test, "classification","xtest",upscale=False)

input_shape = (x_train.shape[1],1)
convolution_layers = count_convolutions(input_shape, kernel_size)

model = keras.models.Sequential()
model.add(Conv1D(16, kernel_size=kernel_size, activation="selu", strides=2, input_shape=input_shape, padding="same"))
for i in range(convolution_layers):
    model.add(Conv1D(32, kernel_size=kernel_size, activation="selu", strides=2,padding="same"))
model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(128, activation="selu"))
model.add(Dropout(0.5))

model.add(Dense(num_classes, activation='softmax'))
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(0.0005),
              metrics=['accuracy'])
model.summary()
#model.fit(np.expand_dims(x_train, axis=2), y_train,
#          batch_size=batch_size,
#          epochs=epochs,
#          verbose=1,
#          validation_data=(np.expand_dims(x_test, axis=2), y_test))
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))

score = model.evaluate(x_test, y_test, verbose=0)

print('Test loss:', score[0])
print('Test accuracy:', score[1])

model.save(model_savepath)
print('model saved to ', model_savepath)