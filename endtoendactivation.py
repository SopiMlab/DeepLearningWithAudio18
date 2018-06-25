import keras
from pydub import AudioSegment
from keras.models import Sequential, Model
from keras.layers.advanced_activations import LeakyReLU
from keras.layers import Input, Dense, UpSampling1D, Conv1D, Activation, Reshape, BatchNormalization
import numpy as np
from playsound import play_sound, play_and_save_sound

latent_dim = 100
sound = AudioSegment.from_wav("input/speech_commands/bird/0a7c2a8d_nohash_1.wav")
samples = int(sound.frame_count())
sound = sound.set_channels(1) 
soundarray = sound.get_array_of_samples()
target = []
target.append(np.array(soundarray))
target = np.array(target)
print(target.shape)

#target = target.reshape(target.shape[0], target.shape[1], 1)

input_shape = (target.shape[1],1)

play_sound(target, "original")

model = Sequential()

model.add(Activation("selu", input_shape=input_shape))

model.summary()

noise = Input(shape=(samples,))
clip = model(noise)

generator = Model(noise, clip)

gen_clip = generator.predict(target, 1)

play_sound(gen_clip, "generated")