import keras
from pydub import AudioSegment
from keras.models import Sequential, Model
from keras.layers.advanced_activations import LeakyReLU
from keras.layers import Input, Dense, UpSampling1D, Conv1D, Activation, Reshape, BatchNormalization
import numpy as np
from playsound import play_sound, play_and_save_sound
from audio_loader import load_all

latent_dim = 100

#sound = AudioSegment.from_wav("input/speech_commands/bird/0a7c2a8d_nohash_1.wav")
sound = load_all("categorized", "cat",forceLoad=True)
sound = sound / 65536
target = np.array(sound[0])
print(target.shape)

input_shape = (target.shape[1],1)

play_and_save_sound(sound, "endtoend2", "original", upscale=True)

model = Sequential()

model.add(Activation("sigmoid", input_shape=input_shape))

model.summary()

noise = Input(shape=(len(sound[0]),))
clip = model(noise)

generator = Model(noise, clip)

target = target.reshape(target.shape[1], target.shape[0])
gen_clip = generator.predict(target, 1)

play_and_save_sound(gen_clip, "endtoend2", "sigmoidactivated", upscale=True)