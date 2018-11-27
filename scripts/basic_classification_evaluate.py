import sys
sys.path.insert(0, 'tools')

import audio_loader
import keras
import numpy as np
import os, shutil
import split

os.environ["CUDA_VISIBLE_DEVICES"]="0"



sounds_dir = "testnonbeatles"
result_file = os.path.join("log",sounds_dir+"_classification_result_3sec2.txt")
model_savepath = "saved_model_3sec2_2400_40"
split_length = 3
sample_rate = 16384

model = keras.models.load_model(model_savepath)
print("keras model " + model_savepath + " loaded")

songs = os.listdir(sounds_dir)
songs = [os.path.join(sounds_dir,f) for f in os.listdir(sounds_dir) if os.path.isfile(os.path.join(sounds_dir, f))]


print("Model: " + model_savepath,file=open(result_file, "a"))
for song in songs:
    readsong = split.read_and_split_audio(song,split_length, 0)
    audios = audio_loader.segments2array(readsong,sample_rate)
    result = model.predict(audios)
    print(song + ": " + str(result.mean(0)[0]),file=open(result_file, "a"))
    print(song + ": " + str(result.mean(0)[0]))





