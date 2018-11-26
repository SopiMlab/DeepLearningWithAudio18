from pydub import AudioSegment
import sys
import glob
import os
import ntpath

# split audio files into fixed length of audio

def read_and_split_audio(audio_file, split_seconds, offset=0, save_dir = None, save_name = None):
    """
    read the an audio file and split it into segments.
     audio_file: an audio file path
     split_seconds: length of each split
     offset: initial offset before splitting
     save_dir: if you want to save the splitted files, this is the directory to save. None means no save
     save_name: the base name for saving the splitted files. An index will be added after the base name

    Return: the list of splitted AudioSegment objects.

    """
    split_length = float(split_seconds)
    seconds = split_length * 1000

    filename_without_ext = os.path.splitext(ntpath.basename(audio_file))[0]

    song = AudioSegment.from_wav(audio_file)
    clips = []

    if song.duration_seconds <= offset:
        return clips

    if save_dir!= None and not os.path.exists(save_dir):
        os.makedirs(save_dir)

    i = 0

    start_index = offset*1000

    while song.duration_seconds*1000 > start_index + seconds:
        clip = song[start_index:start_index + seconds]
        start_index += seconds
        clips.append(clip)
        if save_dir != None:
            clip.export((os.path.join(save_dir, filename_without_ext + str(i) + ".wav")), format='wav')
            i += 1
    
    return clips



def split_audios(sounds_dir, out_dir, name, split_seconds):
    split_length = float(split_seconds)


    wav_fps = os.listdir(sounds_dir)
    wav_fps = [f for f in os.listdir(sounds_dir) if os.path.isfile(os.path.join(sounds_dir, f))]
    seconds = split_length * 1000
    i = 0

    print("clip count " + str(len(wav_fps)))

    total_length = 0
    total_clips_to_clip = 0
    for wav in wav_fps:

        song = AudioSegment.from_wav(os.path.join(sounds_dir,wav))
        total_length += song.duration_seconds
        total_clips_to_clip += song.duration_seconds // split_length
    
    print("Total length of clips: " + str(total_length))
    print("Splitting to exactly " + str(total_clips_to_clip) + " clips.")

    if not os.path.exists(out_dir):
            os.makedirs(out_dir)

    for wav_fp in wav_fps:
        song = AudioSegment.from_wav(os.path.join(sounds_dir,wav_fp))
        while song.duration_seconds > split_length:
            clip = song[:seconds]
            song = song[seconds:]
            clip.export((os.path.join(out_dir, name + str(i) + ".wav")), format='wav')
            i += 1

    print(str(i) + " files splat")


#readsong = read_and_split_audio(os.path.join("test_audios","Wonderwall.wav"),2, 1,save_dir="testsave")
#print(len(readsong))

def main():
    sounds_dir, out_dir, name, split_seconds = sys.argv[1:5]
    split_audios(sounds_dir, out_dir, name, split_seconds)