import os
from pydub import AudioSegment

sounds_dir = "input/categorized/car_horn" 
wav_fps = os.listdir(sounds_dir)

for wav in wav_fps:
    sound = AudioSegment.from_wav(sounds_dir + wav)
    array = sound.get_array_of_samples()
    print(array)
