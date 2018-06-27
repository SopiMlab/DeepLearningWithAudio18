from audio_tools import count_convolutions
from audio_loader import load_audio
import keras
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv1D 

import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

num_classes = 10

(x_train, y_train), (x_test, y_test) = load_audio("categorized", num_classes, forceLoad=True, framerate=16384)

batch_size = 30
epochs = 50
kernel_size = 5

input_shape = (x_train.shape[1],1)
convolution_layers = count_convolutions(input_shape, kernel_size)

model = keras.models.Sequential()
model.add(Conv1D(16, kernel_size=kernel_size, activation='selu', strides=2, input_shape=input_shape, padding="same"))
for i in range(10):
    model.add(Conv1D(32, kernel_size=kernel_size, activation='selu', strides=2,padding="same"))
model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(32, activation='selu'))
model.add(Dropout(0.5))

model.add(Dense(num_classes, activation='softmax'))
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'])
model.summary()
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))

score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])