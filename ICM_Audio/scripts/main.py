import sys
sys.path.insert(0, 'tools')

from audio_loader import load_audio
from playsound import play_and_save_sound

def check_samples (samples):
    min = 10000
    max = 0
    for x in samples:
        for i in x:
            if(i < min):
                min = i
                print(min)
            if(i > max):
                max = i
                print(max)
    print("max wads " + str(max))
    print("min was :" + str(min))

(x_train, y_train), (x_test, y_test) = load_audio("speech_commands", 10)

#play_and_save_sound(x_train, "test")
check_samples(x_train)

