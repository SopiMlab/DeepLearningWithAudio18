import os
from keras.utils import to_categorical
from pydub import AudioSegment
import numpy as np
import array
from playsound import check_sample, update_soundpath

def load_audio(foldername, num_classes = 10, framerate = 0, forceLoad=False, reshape=True):
    folders_dir = "input/" + foldername
    name = folders_dir[(folders_dir.find("/") + 1):]
    if os.path.isfile("input/saved/" + name + ".npz") and not forceLoad and reshape:
        print("Library already loaded!")
        soundlibrary = np.load("input/saved/" + name + ".npz")
        x_train = (soundlibrary['arr_0'])
        y_train = (soundlibrary['arr_1'])
        x_test = (soundlibrary['arr_2'])
        y_test = (soundlibrary['arr_3'])
        path = (soundlibrary['arr_4'])
        print(path)
        update_soundpath(path)
    else:
        x_train = []
        y_train = []
        x_test = []
        y_test = []
        category_folds = os.listdir(folders_dir)
        for i in range(1,num_classes + 1): #make this more stable, only find folders
            print("Loading soundset number {}".format(i))
            folder = folders_dir + "/" + category_folds[i] + "/"
            wav_fps = os.listdir(folder)
            print("{} sounds in category {}".format(len(wav_fps),category_folds[i]))
            trainamount = int(len(wav_fps) // 1.25)
            for j in range(0,trainamount):
                sound = AudioSegment.from_wav(folder + wav_fps[j])
                if framerate != 0:
                    #print("I WANT TO STAND OUT AND SHOW YOU THE FRAME RATE: " + str(sound.frame_rate))
                    sound = sound.set_frame_rate(framerate) # check frame rate and do this based on that. Silly to hard code.
                sound = sound.set_channels(1) 
                soundarray = sound.get_array_of_samples()
                nparray = np.array(soundarray)
                x_train.append(nparray)
                y_train.append(i - 1)
            for j in range(trainamount,len(wav_fps)):
                sound = AudioSegment.from_wav(folder + wav_fps[j])
                if framerate != 0:
                    sound = sound.set_frame_rate(framerate) # check frame rate and do this based on that. Silly to hard code.
                soundarray = sound.get_array_of_samples()
                nparray = np.array(soundarray)
                x_test.append(nparray)
                y_test.append(i - 1)
        path = folder + wav_fps[0]
        update_soundpath(path)
        # Get longest clip from the data.
        max = 0
        for x in x_train:
            if len(x) > max:
                max = len(x)

        if max < framerate:
            max = framerate

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

        y_train = to_categorical(y_train) # this is a bit weird for tensorflow purposes, consider leaving this part out.
        y_test = to_categorical(y_test) # just give them without setting them one-hot.

        x_train, y_train = shuffleLists(x_train, y_train)
        x_test, y_test = shuffleLists(x_train, y_train)

        if(reshape):
            x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
            x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)

        print('x_train shape:', x_train.shape)
        print('y_train shape:', y_train.shape)
        print(x_train.shape[0], 'train samples')
        print(x_test.shape[0], 'test samples')

        name = folders_dir[(folders_dir.find("/") + 1):]
        print("Saving arrays to file")
        if not os.path.exists("input/saved/"):
            os.makedirs("input/saved/")
        np.savez("input/saved/" + name, x_train,y_train,x_test,y_test, path)
    return (x_train,y_train),(x_test,y_test)

def shuffleLists(a,b):
    indices = np.arange(a.shape[0])
    np.random.shuffle(indices)

    a = a[indices]
    b = b[indices]
    return a,b

def load_all(foldername, categoryname ="",framerate = 0, forceLoad=False, reshape=True):
    folders_dir = "input/" + foldername + "/" + categoryname
    name = foldername + categoryname
    if os.path.isfile("input/saved/" + name + ".npz") and not forceLoad and reshape:
        print("Library already loaded!")
        soundlibrary = np.load("input/saved/" + name + ".npz")
        x_train = (soundlibrary['arr_0'])
        path = (soundlibrary['arr_1'])
        print(path)
        update_soundpath(path)
    else:
        x_train = []
        wavs = os.listdir(folders_dir)
        for wav in wavs:
            sound = AudioSegment.from_wav(folders_dir + "/" + wav)
            if framerate != 0:
                sound = sound.set_frame_rate(framerate) # check frame rate and do this based on that. Silly to hard code.
                #sprint("I WANT TO STAND OUT AND SHOW YOU THE FRAME RATE: " + str(sound.frame_rate*sound.duration_seconds))
            sound = sound.set_channels(1) 
            soundarray = sound.get_array_of_samples()
            nparray = np.array(soundarray)
            x_train.append(nparray)

        path = folders_dir + "/" + wavs[0]
        update_soundpath(path)
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

        if(reshape):
            x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)

        print('x_train shape:', x_train.shape)
        print(x_train.shape[0], 'train samples')

        # for x in x_train:
        #     check_sample(x)

        print("Saving arrays to file")
        if not os.path.exists("input/saved/"):
            os.makedirs("input/saved/")
        np.savez("input/saved/" + name, x_train, path)
    return x_train