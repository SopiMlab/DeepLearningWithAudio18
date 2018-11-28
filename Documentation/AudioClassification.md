# Audio Classification
## NOTE
The training if much more faster if you have a powerful Nvdia GPU.

By complete this tutorial, you will be able to train a model to predict the class of a sound file.

You need to do some installing and things might just break randomly, so prepare to take a few hours just to get things running. 

You need Python 3, Tensorflow and Keras library to train the neural network model. See this [intruction](https://keras.io/#installation) for installing Tensorflow and Keras. Tensorflow 1.8 and Keras 2.2 is recommended now.

If you want to train your model on GPU, you need CUDA, cuDNN, Python 3 and tensorflow-gpu installed, you need to install those. [Instructions here](https://docs.nvidia.com/deeplearning/sdk/cudnn-install/index.html) (When installing CUDA, uncheck Visual Studio Integration, or the installation will fail). 

If you get missing Python module announcement on terminal during any step below, such as "no module named tqqm", run pip install tqqm.

## Training
- Clone this repository

- Prepare the training data. If you already have a dataset of audios of different categories(classes) with fixed length, you can skip this step:

  -- Collect your audios of different categories. Put them into different folders under `scripts/tools` directory of this repository(one folder for each category).
  
  -- Split the audio files into clips with fixed length using split.py script. Open Terminal and navigate to `scripts/tools` directory, run the following command for each of your categories
  
```
python split.py audio_directory output_directory output_basefilename clip_length offset
```
Where the `audio_directory` if the category folder that contains all the audio files of that category, `output_directory` is a folder where you want to put the splitted audio clips, `output_basefilename` is the base name for all splitted audio files, `clip_length` is how long in seconds you want to split the audios into(recommend 3), `offset` is the initial offset(0 is ok)

Use all of your category folders as `audio_directory`, and generate splitted audio files into different for all categories.

After this step, you should have a bunch of folders, each of which has all audio files of that category, with fixed clip length.

- Train the neural network model

 -- Put your training data to the correct place. All training data should be under `input/` folder of this repository. Create a folder of your own name under `input/`, and put all the folders of the splitted audios of different categories under the folder you just created. 
 -- Run the python script for training. Open Terminal and navigate to `scripts` directory,run the following command to start training.
 

* Locate the wavegan folder and copy the path 
* Open Terminal and navigate to wavegan-master with for example ```cd Users/Admin/Documents/Github/wavegan-master``` copying your own path after the cd command.
```
python basic_classification.py dataset_folder number_of_classes model_save_name training_epoch batch_size sample_limit
```
Where `dataset_folder` is the folder you just created that contains all the training data, `number_of_classes` is how many classes to train(less or equal to the category number your dataset contains),`model_save_name` is the name to save the trained model, `training_epoch` is how long to train, `batch_size` is the mini batch size for training(128 is usually good), and sample_limit is how many audio clips for each category will be used for training at most.

If everything is working correctly, you should start seeing something lie this:
![running wavegan, generator vars](images/runclassification.jpg)

When the training is done, you should see the training result in the terminal. The trained model will be saved as a file with the name you specified before.

### Use your trained model to classify audios
