import os
import numpy as np
import array
from pydub import AudioSegment
from pydub.playback import play
import keras
from keras.utils import to_categorical
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv1D

folders_dir = "input/speech_commands"
sounds_dir = "input/speech_commands/"
#folders_dir = "input/categorized"
#sounds_dir = "input/categorized/" 
num_classes = 10

# Replace with:
# (xtrain, ytrain), (x_test, y_test) = audio.load_data()
## LOADING DATA STARTS HERE, MOVE TO ANOTHER FILE
x_train = []
y_train = []
x_test = []
y_test = []

category_folds = os.listdir(folders_dir)
for i in range(1,num_classes + 1):
    print("Loading soundset number {}".format(i))
    folder = sounds_dir + category_folds[i] + "/"
    wav_fps = os.listdir(folder)
    print("{} sounds in category {}".format(len(wav_fps),category_folds[i-1]))
    trainamount = int(len(wav_fps) // 1.25)
    for j in range(0,trainamount):
        sound = AudioSegment.from_wav(folder + wav_fps[j])
        sound = sound.set_frame_rate(11025) # check frame rate and do this based on that. Silly to hard code.
        sound = sound.set_channels(1) 
        soundarray = sound.get_array_of_samples()
        nparray = np.array(soundarray)
        x_train.append(nparray)
        y_train.append(i - 1)
    for j in range(trainamount,len(wav_fps)):
        sound = AudioSegment.from_wav(folder + wav_fps[j])
        sound = sound.set_frame_rate(11025)
        sound = sound.set_channels(1)
        soundarray = sound.get_array_of_samples()
        nparray = np.array(soundarray)
        x_test.append(nparray)
        y_test.append(i - 1)

# Get longest clip from the data.
max = 0
for x in x_train:
    if len(x) > max:
        max = len(x)

# Pad data with zeroes so that all clips are the same length for convolution
new_x_train = []
for x in x_train:
    if len(x) < max:
        x = np.pad(x, (0, max-len(x)), mode='constant')
    new_x_train.append(x)
x_train = np.array(new_x_train)

new_x_test = []
for x in x_test:
    if len(x) < max:
        x = np.pad(x, (0, max-len(x)), mode='constant')
    new_x_test.append(x)
x_test = np.array(new_x_test)

y_train = to_categorical(y_train)
y_test = to_categorical(y_test) 

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)

input_shape = (x_train.shape[1],1)

print('x_train shape:', x_train.shape)
print('y_train shape:', y_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

## DATA LOADING ENDS

batch_size = 30
epochs = 50
kernel_size = 5
# Figure out how many convolutions we can do with custom data, alternatively, let students set this, or do it by hand.
count = 0
x = input_shape[0]
while x > kernel_size:
    x = x/2
    count += 1

model = keras.models.Sequential()
model.add(Conv1D(16, kernel_size=kernel_size, activation='selu', strides=2, input_shape=input_shape))
for i in range(count - 3): #Add Enough convolutions to get close to kernel size
    model.add(Conv1D(32, kernel_size=kernel_size, activation='selu', strides=2))
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
