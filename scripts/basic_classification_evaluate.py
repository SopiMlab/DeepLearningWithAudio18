import sys
sys.path.insert(0, 'tools')

from audio_tools import count_convolutions
from audio_loader import load_audio
import keras
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv1D 
from playsound import save_sound
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

model_savepath = "saved_model"

model = keras.models.load_model(model_savepath)


