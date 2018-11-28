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

#num_classes = 2
#model_savepath = "saved_model_3sec2"
#training_data_folder = "beatles3sec2"

"""
This is the script to train a basic audio classification model.

Usage in command window: python basic_classification.py training_data_folder num_classes model_savepath training_epoch batch_size amount_limit


training_data_folder: the data folder that contains the training data. It needs to be under../input/ folder. 
    Audio clips of different classes need to be in seperate folders under the data folder.All audio clips needs to have exactly the same length
num_classes: How many classes should be used to train. It needs to be less than the classes number contained in the data folder
model_savepath: where the final trained model is saved and its name
training_epoch: how many epochs to train the model.
batch_size: training minibatch size. 128 should be fine usually
amount_limit: how many audio clips to load for each class at most.

"""

def train_basic_audio_classifier(num_classes, model_savepath, training_data_folder,training_epoch, batch_size = 128, amount_limit=5000, forceLoad = True):
"""
train a basic audio classification model.
"""

    epochs = training_epoch
    kernel_size = 5
    framerate = 16384

    (x_train, y_train), (x_test, y_test) = load_audio(training_data_folder, num_classes, forceLoad=forceLoad, framerate=framerate,amount_limit=amount_limit)

    print('x_train shape:', x_train.shape)
    print('x_test shape:', x_test.shape)



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



if __name__ == "__main__":
    training_data_folder, num_classes, model_savepath, training_epoch,batch_size, amount_limit = sys.argv[1:7]
    train_basic_audio_classifier(int(num_classes), model_savepath, training_data_folder, int(training_epoch),int(batch_size),int(amount_limit))
