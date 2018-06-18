import os
import numpy as np
import array
from pydub import AudioSegment
from pydub.playback import play
import keras
from keras.utils import to_categorical
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv1D

folders_dir = "input/categorized"
sounds_dir = "input/categorized/" 
# folders_dir = "input/speech_commands"
# sounds_dir = "input/speech_commands/"
category_folds = os.listdir(folders_dir)
x_train = []
y_train = []
x_test = []
y_test = []


for i in range(1,11):
    #print("Loading soundset number {}".format(i))
    folder = sounds_dir + category_folds[i] + "/"
    wav_fps = os.listdir(folder)
    #print("{} sounds in category {}".format(len(wav_fps),i))
    trainamount = int(len(wav_fps) // 1.25)
    #print("{} to test, rest to train".format(testamount))
    for j in range(0,trainamount): ## CHANGE THIS TO TAKE ONLY PART OF THE DATA
        #print(wav_fps[j])
        sound = AudioSegment.from_wav(folder + wav_fps[j])
        sound = sound.set_frame_rate(11025)
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

print("{} sounds to train with".format(len(x_train)))
#print("in {} categories ".format(num_classes))


# sound = AudioSegment.from_file('input/speech_commands/bed/1bb574f9_nohash_0.wav')
# print("{} is the frame rate.".format(sound.frame_rate))
# print(array)
# sound = sound.set_channels(1)
# print("{} is the frame rate as mono.".format(sound.frame_rate))
# print(x_train[150])
# shifted_samples_array = array.array(sound.array_type, x_train[8])
# new_sound = sound._spawn(shifted_samples_array)

# print("playing sound from category " + category_folds[y_train[8] + 1])
# play(new_sound)

# for label in category_folds:
#     print(label) 

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

x_train = np.array(x_train)
y_train = np.array(y_train)
x_test = np.array(x_test)
y_test = np.array(y_test)

#x_train shape: (60000, 28, 28, 1)
# should be : (320, 22050, 1)

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)

input_shape = (x_train.shape[1],1)

print('x_train shape:', x_train.shape)
print('y_train shape:', y_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

batch_size = 3
epochs = 20
num_classes = 10

model = keras.models.Sequential()
#model.add(Dense(100, activation='relu', input_shape=x_train.shape))
#model.add(Dense(50, activation='relu'))
#model.add(Dense(num_classes, activation='softmax',input_shape=x_train.shape))
model.add(Conv1D(16, kernel_size=5, activation='relu', strides=2, input_shape=input_shape))
model.add(Conv1D(32, kernel_size=5, activation='relu', strides=2))
model.add(Conv1D(32, kernel_size=5, activation='relu', strides=2))
model.add(Conv1D(32, kernel_size=5, activation='relu', strides=2))
model.add(Conv1D(32, kernel_size=5, activation='relu', strides=2))
model.add(Conv1D(32, kernel_size=5, activation='relu', strides=2))
model.add(Conv1D(32, kernel_size=5, activation='relu', strides=2))
model.add(Conv1D(32, kernel_size=5, activation='relu', strides=2))
model.add(Conv1D(32, kernel_size=5, activation='relu', strides=2))
model.add(Conv1D(32, kernel_size=5, activation='relu', strides=2))
model.add(Conv1D(32, kernel_size=5, activation='relu', strides=2))
model.add(Conv1D(32, kernel_size=5, activation='relu', strides=2))


#model.add(Dense(10, activation='relu'))
#model.add(Conv1D(32, 5, activation='relu', strides=2))
#model.add(Conv1D(32,5, activation='relu', strides=2))
#model.add(Conv1D(32,5, activation='relu', strides=2))
#model.add(Conv1D(32,5, activation='relu', strides=2))
model.add(Flatten())
#model.add(Dense(32, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(lr=0.0001),
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







# Make a train, test split, with 80:20?
# Though need a lot of different examples to make a relevant test case.
# also create Y, for ground truth.
