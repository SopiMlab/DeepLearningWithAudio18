''' This script splits all the files in a folder to folders named 'train', 'valid' and 'test' with a 80-10-10 split.'''

import sys
import os

sounds_dir = sys.argv[1]

if("input" not in sounds_dir):
    print("invalid path, aborting")
    sys.exit()

files = os.listdir(sounds_dir)
i = 0

print("clip count " + str(len(files)))

total_clips = len(files)
    
trainamount = total_clips // 1.25
validamount = total_clips // 10
testamount = total_clips // 10

extraclips = total_clips % 5

print("Train: " + str(total_clips // 1.25))
print("Valid and Test " + str(total_clips // 10) + " each")

if not os.path.exists(sounds_dir + "/train"):
    os.makedirs(sounds_dir + "/train")
if not os.path.exists(sounds_dir + "/valid"):
    os.makedirs(sounds_dir + "/valid")
if not os.path.exists(sounds_dir + "/test"):
    os.makedirs(sounds_dir + "/test")

i = 0

for j in range(0,int(trainamount)):
    os.rename(sounds_dir + "/" + files[i], sounds_dir + "/train/" + files[i])
    i += 1

for j in range(0,int(validamount)):
    os.rename(sounds_dir + "/" + files[i], sounds_dir + "/valid/" + files[i])
    i += 1

for j in range(0,int(testamount)):
    os.rename(sounds_dir + "/" + files[i], sounds_dir + "/test/" + files[i])
    i += 1

print(str(i) + " files sorted into train, valid and test")