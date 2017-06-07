# TubeCam

## Project structure
|Folder|Description|
|---|---|
|Analytics|Contains analytical tools helping out to understand the model. These are written and/or used in jupyter notebooks format. Following were used:<br><ul><li>t-SNE Visualization</li><li>DBSCAN-Clusters</li><li>Confusion Matrix</li><li>Learning Curve</li><li>Accuracy</li><li>Lime</li></ul>|
|Data processing|Scripts to generate images out of a video. There is another script to delete duplicates|
|Library|Collection of shared scripts, which can later be included in any other scripts|
|Resources|All Resources like pictures for training and testing (due to licence constraints, no example given!)|
|Training|Scripts for training a model. Contains also the resulting model as a frozen graph and logs, which are generated during training|
<br>

## Dependencies
Dependencies in notebooks are listed as `!pip install` commands, which are markup fields. If any dependencies are missing on your system, change those field to code-fields and execute.

<br>

## Attachments
Due to size constraints, some files had to be uploaded somewhere else.

|File|URL|
|---|---|
|Logs|https://1drv.ms/f/s!Arr4EpAOX-48hsJvxN72YJ7-yxbPJg|
|Models|https://1drv.ms/f/s!Arr4EpAOX-48hsJqaOLI4TvTL5fCTA|

<br>

## Data processing
The following two console-based scripts are available for data processing:
* `createImageFromMotion.py`
* `duplicateRemover.py`

### Detecting motion and create images of it
#### Overview
Using OpenCV a given video file will be analyzed for motion frame-by-frame. If any motion is found it will be cropped rectangular and exported as a file. If multiple moving spots were found, only the largest one will be saved.

Parts of this script are based on the "Basic motion detection and tracking with Python and OpenCV"
by Adrian Rosebrock. Available at:
http://www.pyimagesearch.com/2015/05/25/basic-motion-detection-and-tracking-with-python-and-opencv/

#### Usage
The script `createImageFromMotion.py` only needs a single parameter to work:
* `-p` or `--path`, path to the folder containing your videos

#### Cleaning up duplicates
The creation of images using the `createImageFromMotion.py` script can lead to an excessive amount of new data This is due to the fact that the parameters of the script are tradeoffs to fit the most animals given during creation of this work. To counter this the script `duplicateRemover.py` can be used as follows:

* `-p` or `--path`, path to root folder of the images that should be cleaned up

## Training
Retrain was done using the official TensorFlow `retrain.py` example available at:


The most important parameter for usage are:

* `--bottleneck_dir=/tf_files/bottlenecks`, path where the bottleneck files will be stored
* `--how_many_training_steps 750`, the amount of training steps
* `--model_dir=/tf_files/inception`, base model to be used
* `--output_graph=/tf_files/retrained_graph.pb`, path where the retrained graph will be stored
* `--output_labels=/tf_files/retrained_labels.txt`, path where the retrained labels will be stored
* `--random_crop=5`, data augmentation 5% random cropping
* `--image_dir ./Resources/Pictures/Training/`, path to picture training set

<br>

## Analytics
### Classification of unknown animals
The file `unknownAnimal.py` can be used to classify unknown animals fast and easy. The result will be a list containing the names of the recognized animals.
The code of this file is nearly identical to the one in the Jupyter notebook `Tensorboard.ipynb` found in the Analytics folder.
To use this file, the following parameters must be given:

* `-p` or `--path`, path to the folder of images that should be classified
* `-m` or `--model`, path to the model for the classification
* `-l` or `--label`, path to the labels used in the model