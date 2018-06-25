from pydub import AudioSegment
from pydub.playback import play
import array
import time

def play_sound(samples, label): # We don't know what the original file was like at this point anymore. AKA length and framerate. This works for now
    sound = AudioSegment.from_file('input/speech_commands/bed/1bb574f9_nohash_0.wav')
    shifted_samples_array = array.array(sound.array_type, samples)
    new_sound = sound._spawn(shifted_samples_array)
    print("playing sound from category " + str(label))
    play(new_sound)

def play_and_save_sound(samples, label):
    check_sample(samples[1])
    sound = AudioSegment.from_file('input/speech_commands/bed/1bb574f9_nohash_0.wav')
    shifted_samples_array = array.array(sound.array_type, upscale_sample(samples[1]))
    new_sound = sound._spawn(shifted_samples_array)
    print("playing and saving sound from category " + str(label))
    play(new_sound)
    new_sound.export("output/" + label + str(time.time()) + ".wav", format="wav")

def save_sound(samples, label):
    sound = AudioSegment.from_file('input/speech_commands/bed/1bb574f9_nohash_0.wav')
    shifted_samples_array = array.array(sound.array_type, samples)
    new_sound = sound._spawn(shifted_samples_array)
    name = label + str(time.time()) + ".wav"
    print("saving sound as " + name)
    new_sound.export("output/" + name, format="wav")

def check_sample (sample):
    min = 10000
    max = 0
    for i in sample:
        if(i < min):
            min = i
        if(i > max):
            max = i
    print("max wads " + str(max))
    print("min was :" + str(min))
    
def upscale_sample(sample):
    check_sample(sample * 16000)
    return sample * 16000