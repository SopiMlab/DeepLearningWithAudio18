import sys
import glob
import os

#sounds_dir, argh = sys.argv[1:2]

#This script sorts the massive NSYNTH dataset (https://magenta.tensorflow.org/datasets/nsynth#files) from one folder to multiple folders based on instrument.
#Just move this script to the same folder as the nsynth audio-folder.

sounds_dir = "audio/"

if not os.path.exists('nsynth/'):
	os.makedirs('nsynth/')

files = os.listdir(sounds_dir)
print("clip count " + str(len(files)))

for file in files:
	find = file.find(".wav")
	category = (file[:(find-12)])
	print(category)
	if not os.path.exists("nsynth/" + category):
		os.makedirs('nsynth/' + category)
		print("making dir")
	os.rename('audio/' + file,'nsynth/' + category + '/' +  file)
