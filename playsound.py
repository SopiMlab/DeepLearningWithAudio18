from pydub import AudioSegment
from pydub.playback import play
import array

def play_sound(samples, label): # We don't know what the original file was like at this point anymore. AKA length and framerate. This works for now
    sound = AudioSegment.from_file('input/speech_commands/bed/1bb574f9_nohash_0.wav')
    shifted_samples_array = array.array(sound.array_type, samples)
    new_sound = sound._spawn(shifted_samples_array)
    print("playing sound from category " + str(label))
    play(new_sound)