# Deep Learning with Audio 

DOM-E5129 - Intelligent Computational Media

<!-- ## State of Audio Deep Learning -->


### State of audio generation in Deep Learning (December 2018)
Speech and music (MIDI) generation are doing well, however the methods that work well with images don’t translate that well to the audio domain. Turning sounds to spectrograms and different signal processing algorithms make it possible to use image models, but the results tend to be a bit underwhelming and the sound quality is bad.

[A blog post going deeper into why this is the case.](https://towardsdatascience.com/whats-wrong-with-spectrograms-and-cnns-for-audio-processing-311377d7ccd)

WaveNet (September 2016) was a massive breakthrough in audio generation. It creates waveforms sample by sample which seems to be the reason why it generates so much better results. It’s a convolutional neural network that wasn’t usually used for generation before. It is mainly used to create natural speech, but there was some tests with music generation too. This is one of the applications that has seen widespread real-world use.

[Two Minute Paper video about WaveNet](https://www.youtube.com/watch?v=CqFIVCD1WWo) 

[Continuation Paper that makes generation a lot faster](https://arxiv.org/pdf/1711.10433.pdf) (November 2017)

[It's part of Google Duplex, the restaurant reservation Assistant](https://ai.googleblog.com/2018/05/duplex-ai-system-for-natural-conversation.html) (May 2018)

The case of GANs is a good case to show how the audio domain is progressing is a lot slower than computer vision and image generation. Considering the original GAN paper came out in 2014 and there’s multiple amazing applications of it in the recent years. It took until 2018 until anyone managed to combine WaveNet sample generation approach and GAN.

[Failed attempt from January 2017](http://deepsound.io/dcgan_spectrograms.html)

[Successful version from January 2018](http://deepsound.io/pggan_specs.html)

One of the most promising works is [“A Universal Music Translation Network”](https://research.fb.com/publications/a-universal-music-translation-network/) (May 2018) by Facebook Research. It can take a piece of music played one way and translate it to another style. Piano -> Harpsichord, Band -> Orchestra, Whistling -> Orchestra. It uses a clever system of convolving input into a shared musical “language” that it can then translate to different styles or instruments with separately trained models. Unfortunately the code for the project is not available and trained for 6 days with 8 GPUs.

One huge problem with all of these system is that the results are very idealised, when you pick only the best results, it gives a misleading picture of what is actually possible.
Good early example is [GRUV](https://github.com/MattVitelli/GRUV) all the way from 2015.
It seems it could generate music, but it actually just memorizes it [(down to the lyrics)](https://youtu.be/0VTI1BBLydE?t=3m36s).
A more likely scenario in the current situation is presented in [this video](https://www.youtube.com/watch?v=dTYdRX1b000) (three full days of training with just some plausible stuttering backing vocals to show.)

With massive datasets, the likelihood of your impressive results being just clever sampling from the dataset seems very likely.

The only reasonable and accessible system seems to be [Magenta](https://magenta.tensorflow.org/).
It has a great set of trained models for different types of musical improvisation.
It is also designed to work on the browser for [fun](https://codepen.io/teropa/full/RMGxOQ/), [easily accessible](https://codepen.io/iansimon/full/Bxgbgz/) [toys](https://experiments.withgoogle.com/ai/ai-duet/view/).
The problem is that it’s mainly MIDI-based, which massively limits the possibilities.
Magenta also includes [NSynth](https://experiments.withgoogle.com/ai/sound-maker/view/), a system that can combine instruments in fascinating ways.
And you can actually [use it as an instrument](https://www.youtube.com/watch?v=0fjopD87pyw) (March 2018).

Almost all of the applications listed here take intense amounts of training.
Most of the big papers are training with 10-32 GPUs for around a week. 

So any attempted practical application of these systems is likely to be unsuccessful at the current time.
### Promising or interesting works
* [WaveGAN](https://github.com/chrisdonahue/wavegan) (February 2018)

* [SampleRNN](https://arxiv.org/pdf/1612.07837.pdf) (February 2017)
  * [A video training with pop music](https://www.youtube.com/watch?v=dTYdRX1b000) (March 2018)

* [A Universal Music Translation Network](https://research.fb.com/publications/a-universal-music-translation-network/) (May 2018)

* [Time-Domain audio style transfer](https://github.com/pkmital/time-domain-neural-audio-style-transfer) (November 2017)

* [Magenta](https://magenta.tensorflow.org/) (2017->)
  * [NSynth](https://magenta.tensorflow.org/nsynth) 
  
* [Creating sounds for silent videos](http://bvision11.cs.unc.edu/bigpen/yipin/visual2sound_webpage/visual2sound.html) (December 2017)
  * This one uses SampleRNN and heavily curated Google Audioset to create plausible sound effects for videos.

* [RunwayML](https://runwayml.com/) (Upcoming)
  * Basically aiming to do what the course is doing. Though even more focus on removing barriers, going as far as not needing to know almost anything about machine learning.
### Strange and interesting offshoot work
* [Three-armed robot drummer](http://www.news.gatech.edu/2016/02/17/wearable-robot-transforms-musicians-three-armed-drummers) (February 2016)

* [Placing plausible sounds on a silent video](http://vis.csail.mit.edu/) (April 2016. The system doesn’t create new sounds, it just picks the most appropriate sound from it’s database)

### Datasets
This is also one huge problem currently. There isn’t many high-quality large audio datasets. Especially for non-music, non-speech sounds, it feels pretty dead.

* [Google AudioSet](https://research.google.com/audioset/)
  * Is really big and categorized, but the problem is that it’s just 10-second clips of Youtube videos, with the type of sound somewhere in there. And one clip might even multiple types of sound. Good for classification, terrible for generation. Also, there's some legal problems of getting just the audio from these videos.
  * [The VEGAS dataset](http://bvision11.cs.unc.edu/bigpen/yipin/visual2sound_webpage/visual2sound.html) is a human-curated subset of AudioSet that is less noisy and generally better for sound generation tasks.

* [ESC-50](https://github.com/karoldvl/ESC-50)
  * A Dataset of 50-different environmental sounds. It’s main use is benchmarking classification, but it’s one of the only sources of environmental quality sounds currently. The problem is that it’s very small, 40 sounds per category. Makes it tricky to use for generation.

* [The NSynth Dataset](https://magenta.tensorflow.org/datasets/nsynth)
  * Absolutely massive set of 300 000 sound files. It’s basically notes played on different instruments. It’s done with MIDI instruments, so not the most interesting form that sense, but it’s easily big enough for generation too

* [Speech Commands Dataset](https://ai.googleblog.com/2017/08/launching-speech-commands-dataset.html) and [SC Zero to Nine Speech Commands](http://deepyeti.ucsd.edu/cdonahue/sc09.tar.gz) 
  * There’s multiple datasets for speech commands and they tend to be large and high-quality. Human speech is just not the most interesting thing to generate, but it’ll likely be the baseline for any future systems. 

* [Kaggle audio datasets](https://www.kaggle.com/datasets?search=audio)
  * There's some strange things here and more must be coming, but the quality varies wildly.

* There’s also many sources of sound effects for example, but considering the amount you need, collecting them from different sources would be a major undertaking.
One fun one is [the BBC sound effect archive.](http://bbcsfx.acropolis.org.uk/)

### Other notes

* The audio sample approach is so unexplored that many frameworks don’t even have a Conv1DTranspose-implementation. So people make their own by running it through Conv2DTranspose.
* The only [audio tutorial](https://www.tensorflow.org/tutorials/sequences/audio_recognition) for Tensorflow is based on spectrograms and only does speech recognition.
### Other interesting links
* [Creative.ai](https://medium.com/@creativeai/creativeai-9d4b2346faf3)
  * An organization dedicated to creating interesting creative applications of AI in as many different fields as possible.  
* [Keras-GAN on GitHub](https://github.com/eriklindernoren/Keras-GAN)
  * Repository of most of the biggest image GANs, implemented in Keras.
* [SeedBank](http://tools.google.com/seedbank/)
  * A collection of Interactive Machine learning examples running on Google CoLab (with free GPUs)


<!-- --------------------------------------------------------------------------------------------------

These are documentation files that are readable withing Github. Just click one and read.

* ```StateOfAudioML.md``` is about the current state and challenges of machine learning projects in the audio domain.
* ```UsingWavegan.md``` is a step-by-step guide on how to use [WaveGAN](https://github.com/chrisdonahue/wavegan).
* ```wavegantools``` is a folder for useful scripts to use with WaveGAN.
* ```images``` are just the images used in the documentation.
-->





# Deep Learning With Audio
- Documentation on different Deep Learning Audio systems as well as instructions on using some of them.
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
