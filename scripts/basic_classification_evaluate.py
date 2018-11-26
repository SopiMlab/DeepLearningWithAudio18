import sys
sys.path.insert(0, 'tools')

import audio_loader
import keras
import numpy as np
import os, shutil
import split

os.environ["CUDA_VISIBLE_DEVICES"]="0"


result_file = "classification_result.txt"
sounds_dir = "test_audios"
model_savepath = "saved_model_20"
split_length = 1.5
sample_rate = 16384

model = keras.models.load_model(model_savepath)
print("keras model " + model_savepath + " loaded")

songs = os.listdir(sounds_dir)
songs = [os.path.join(sounds_dir,f) for f in os.listdir(sounds_dir) if os.path.isfile(os.path.join(sounds_dir, f))]

for song in songs:
    readsong = split.read_and_split_audio(song,split_length, 0)
    audios = audio_loader.segments2array(readsong,sample_rate)
    result = model.predict(audios)
    print(song + ": " + str(result.mean(0)[0]),file=open(result_file, "a"))
    print(song + ": " + str(result.mean(0)[0]))





