# Machine Learning With Audio
- Documentation on different Machine Learning Audio systems as well as instructions on using some of them.
- Tools for loading, playing and plotting audio.
- Some working simple classifiers
- Non-working sample-level/raw audio GANs
- Python scripts for sorting different popular datasets

All of these are meant to work as starting points or aids for audio machine learning, but many of the examples are still very rudimentary and might not produce any meaningful results.

# Tested data sets
- [ESC-50](https://github.com/karoldvl/ESC-50) (Use ```separate_files.py``` to sort into folders)
- [The Speech Commands dataset](https://storage.cloud.google.com/download.tensorflow.org/data/speech_commands_v0.02.tar.gz)
- [NSYNTH dataset](https://magenta.tensorflow.org/datasets/nsynth) (Use ```sort_nsynth.py``` to sort into folders)

The loader assumes that any data is put into a folder called input. You should create one and put any audio you have in organized folders within.
