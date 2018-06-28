from pydub import AudioSegment
from pydub.playback import play
import matplotlib.pyplot as plt
import array
import time
import os

def play_sound(sample, label, upscale=False): # We don't know what the original file was like at this point anymore. AKA length and framerate. This works for now
    #sound = AudioSegment.from_file('input/speech_commands/bed/1bb574f9_nohash_0.wav')
    sound = AudioSegment.from_file('input/categorized/cat/1-34094-A-5.wav')
    playsound = sample[0]
    if upscale:
        playsound = upscale_sample(playsound)
    shifted_samples_array = array.array(sound.array_type, playsound)
    new_sound = sound._spawn(shifted_samples_array)
    print("playing sound from category " + str(label))
    play(new_sound)

def play_and_save_sound(samples, label, run_name="", epoch=0, upscale=True):
    check_sample(samples[0])
    #sound = AudioSegment.from_file('input/speech_commands/bed/1bb574f9_nohash_0.wav')
    sound = AudioSegment.from_file('input/categorized/cat/1-34094-A-5.wav')
    sound.set_channels(1)
    print(sound.array_type)
    playsound = samples[0]
    if upscale:
        playsound = upscale_sample(playsound)
    sample_array = array.array(sound.array_type, playsound)
    new_sound = sound._spawn(sample_array)
    if not os.path.exists("output/" + label + "/"):
        os.makedirs("output/" + label + "/")
    #print(sample_array)
    plt.figure(figsize=(30,10))
    plt.plot(sample_array)
    plt.savefig("output/" + label + "/" + run_name + "#" + str(epoch))
    plt.clf()
    plt.cla()
    print("playing and saving sound from category " + str(label))
    play(new_sound)
    new_sound.export("output/" + label + "/" + run_name + "#" + str(epoch) + ".wav", format="wav")

def save_sound(sample, label):
    sound = AudioSegment.from_file('input/speech_commands/bed/1bb574f9_nohash_0.wav')
    shifted_samples_array = array.array(sound.array_type, sample)
    new_sound = sound._spawn(shifted_samples_array)
    name = label + str(time.time()) + ".wav"
    print("saving sound as " + name)
    new_sound.export("output/" + name, format="wav")

def check_sample (sample):
    mini = 10000
    maxi = 0
    for i in sample:
        if(i < mini):
            mini = i
        if(i > maxi):
            maxi = i
    print("max was " + str(maxi))
    print("min was " + str(mini))

def check_scale (sample):
    mini = 10000
    maxi = 0
    for i in sample:
        if(i < mini):
            mini = i
        if(i > maxi):
            maxi = i
    return max(abs(maxi), abs(mini))
    
def upscale_sample(sample): 
    check_sample(sample)
    scale = check_scale(sample)
    print("scale {}".format(scale))
    if(scale > 0.5): # this is a terrible approach that makes no sense. Also it shouldn't get over the limit anyway. But it does, so no wonder the GAN goes nuts.
        new_sample = (sample /scale) * 32768
        print("went over 1, this should be impossible")
    else:
        new_sample = sample * 65536
    check_sample(new_sample)
    return new_sample.astype(int)