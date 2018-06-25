import keras
from pydub import AudioSegment
from keras.models import Sequential, Model
from keras.layers.advanced_activations import LeakyReLU
from keras.layers import Input, Dense, Reshape, Flatten, BatchNormalization
import numpy as np
from playsound import play_sound, play_and_save_sound

latent_dim = 100
sound = AudioSegment.from_wav("input/speech_commands/bird/0a7c2a8d_nohash_1.wav")
samples = int(sound.frame_count())
sound = sound.set_channels(1) 
soundarray = sound.get_array_of_samples()
target = []
target.append(np.array(soundarray))
#print(target.shape)
target = np.array(target)
print(target.shape)

target = target.reshape(target.shape[0], target.shape[1], 1)

play_sound(target, "original")

model = Sequential()

model.add(Dense(256, input_dim=latent_dim))
model.add(LeakyReLU(alpha=0.2))
model.add(BatchNormalization(momentum=0.8))
model.add(Dense(512))
model.add(LeakyReLU(alpha=0.2))
model.add(BatchNormalization(momentum=0.8))
model.add(Dense(1024))
model.add(LeakyReLU(alpha=0.2))
model.add(BatchNormalization(momentum=0.8))
model.add(Dense(samples, activation='tanh'))
model.add(Reshape([samples,1]))

model.summary()

noise = Input(shape=(latent_dim,))
clip = model(noise)

generator = Model(noise, clip)

noise = np.random.normal(0, 1, (1, latent_dim))
gen_clip = generator.predict(noise, 1)

play_and_save_sound(gen_clip, "generated")

model.compile(loss=keras.losses.mean_squared_error,
              optimizer=keras.optimizers.Adam())

for i in range(100):
    #gen_clip = generator.predict(noise, 1
    print("training: " + str(model.train_on_batch(noise,target)))
    if i % 5 == 0:
        gen_clip = generator.predict(noise, 1)
        play_and_save_sound(gen_clip, "generated")