from pydub import AudioSegment
from pydub.playback import play
import matplotlib.pyplot as plt
import array
import time
import os
import numpy as np
import pandas as pd

plt.rcParams['agg.path.chunksize'] = 10000

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
    plt.ylim(-32768, 32768)
    plt.plot(sample_array)
    plt.savefig("output/" + label + "/" + run_name + "#" + str(epoch))
    plt.clf()
    plt.cla()
    print("playing and saving sound from category " + str(run_name) + " folder " + label)
    play(new_sound)
    new_sound.export("output/" + label + "/" + run_name + "#" + str(epoch) + ".wav", format="wav")

def save_sound(samples, label, run_name="", epoch=0, upscale=True):
    #sound = AudioSegment.from_file('input/speech_commands/bed/1bb574f9_nohash_0.wav')
    sound = AudioSegment.from_file('input/categorized/clapping/1-94036-A-22.wav')
    sound.set_channels(1)
    check_sample(samples[0])
    playsound = samples[0]
    if upscale:
        playsound = upscale_sample(playsound)
    playsound = np.clip(playsound, -32768,32767)
    check_sample(playsound)
    sample_array = array.array(sound.array_type, playsound)
    new_sound = sound._spawn(sample_array)
    if not os.path.exists("output/" + label + "/"):
        os.makedirs("output/" + label + "/")
    #print(sample_array)
    filepath = "output/" + label + "/" + run_name + "#" + str(epoch)
    plot_sound(sample_array,filepath)
    print("saving sound from category " + str(run_name) + " folder " + label)
    new_sound.export(filepath + ".wav", format="wav")

def plot_sound(sample_array, filepath):
    plt.figure(figsize=(30,10))
    plt.ylim(-32768, 32768)
    plt.plot(sample_array)
    plt.savefig(filepath)
    plt.clf()
    plt.cla()
    
def check_sample (sample):
    s = pd.Series(sample.flatten().tolist())
    print(s.describe())

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
    #check_sample(sample)
    new_sample = sample * 65536
    new_sample = new_sample - 32768
    #check_sample(new_sample)
    return new_sample.astype(int)