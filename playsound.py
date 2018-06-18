from pydub import AudioSegment
from pydub.playback import play

sound = AudioSegment.from_file('input/speech_commands/bed/1bb574f9_nohash_0.wav')
print("{} is the frame rate.".format(sound.frame_rate))
print(array)
sound = sound.set_channels(1)
print("{} is the frame rate as mono.".format(sound.frame_rate))
print(x_train[150])
shifted_samples_array = array.array(sound.array_type, x_train[8])
new_sound = sound._spawn(shifted_samples_array)

print("playing sound from category " + category_folds[y_train[8] + 1])
play(new_sound)

for label in category_folds:
    print(label) 