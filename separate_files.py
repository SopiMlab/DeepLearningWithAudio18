import sys
import glob
import os

#sounds_dir, argh = sys.argv[1:2]

# This script sorts the ESC-50 dataset (https://github.com/karoldvl/ESC-50) into folders with meaningful names.

sounds_dir = "audio/"

namedict = {0:'dog', 14:'chirping_birds', 36:'vacuum_cleaner', 19:'thunderstorm', 30:'door_wood_knock', 34:'can_opening', 9:'crow', 22:'clapping', 48:'fireworks', 41:'chainsaw', 47:'airplane', 31:'mouse_click', 17:'pouring_water', 45:'train', 8:'sheep', 15:'water_drops', 46:'church_bells', 37:'clock_alarm', 32:'keyboard_typing', 16:'wind', 25:'footsteps', 4:'frog', 3:'cow', 27:'brushing_teeth', 43:'car_horn', 12:'crackling_fire', 40:'helicopter', 29:'drinking_sipping', 10:'rain', 7:'insects', 26:'laughing', 6:'hen', 44:'engine', 23:'breathing', 20:'crying_baby', 49:'hand_saw', 24:'coughing', 39:'glass_breaking', 28:'snoring', 18:'toilet_flush', 2:'pig', 35:'washing_machine', 38:'clock_tick', 21:'sneezing', 1:'rooster', 11:'sea_waves', 42:'siren', 5:'cat', 33:'door_wood_creaks', 13:'crickets'}

if not os.path.exists('categorized/'):
	os.makedirs('categorized/')

for key, value in namedict.items():
	if not os.path.exists(value):
		os.makedirs('categorized/' + value)
		print("making dir")

files = os.listdir(sounds_dir)
print("clip count " + str(len(files)))

print("category count: " + str(len(namedict)))

for file in files:
	find = file.find(".wav")
	category = (file[find-2:find])
	if category[0] == "-":
		category = category[1]
	os.rename('audio/' + file,'categorized/' + namedict[int(category)] + '/' +  file)
