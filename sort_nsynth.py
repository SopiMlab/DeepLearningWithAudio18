import sys
import glob
import os

#sounds_dir, argh = sys.argv[1:2]

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
