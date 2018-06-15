import os
import tarfile

dest_directory = "speech_commands"
data_url = "https://storage.cloud.google.com/download.tensorflow.org/data/speech_commands_v0.01.tar.gz"

if not os.path.exists(dest_directory):
    os.makedirs(dest_directory)

filepath, _ = urllib.request.urlretrieve(data_url, filepath, _progress)
tarfile.open(filepath, 'r:gz').extractall(dest_directory)
