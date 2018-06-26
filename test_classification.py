from audio_tools import count_convolutions
from audio_loader import load_audio
import keras
from keras.layers import Dense, Dropout, Flatten, LeakyReLU
from keras.layers import Conv1D 

import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

num_classes = 10

(x_train, y_train), (x_test, y_test) = load_audio("speech_commands", num_classes)

batch_size = 30
epochs = 50
kernel_size = 5

input_shape = (x_train.shape[1],1)
convolution_layers = count_convolutions(input_shape, kernel_size)

model = keras.models.Sequential()
model.add(Flatten(input_shape=input_shape))
model.add(Dense(512))
model.add(LeakyReLU(alpha=0.2))
model.add(Dense(256))
model.add(LeakyReLU(alpha=0.2))

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