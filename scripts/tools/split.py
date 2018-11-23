from pydub import AudioSegment
import sys
import glob
import os

# Make this script count the files, only accept wavs,
# and then automatically separate them into train,
# valid and test, with some percentages.
# to really go nuts, make sure that sources are not the same for the buckets.

sounds_dir, out_dir, name, split_seconds = sys.argv[1:5]

split_length = float(split_seconds)

wav_fps = os.listdir(sounds_dir)
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
trainamount = total_clips_to_clip // 1.25
validamount = total_clips_to_clip // 10
testamount = total_clips_to_clip // 10

extraclips = total_clips_to_clip % 5
# print("extraclips " + str(extraclips))
# for j in range(0,3):
# 	print(j)

print("Train: " + str(total_clips_to_clip // 1.25))
print("Valid and Test " + str(total_clips_to_clip // 10) + " each")

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
