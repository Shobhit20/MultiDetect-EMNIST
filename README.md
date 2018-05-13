# MultiDetect-EMNIST
The repository contains the code for detecting multiple characters from an image using the EMNIST dataset

## Dataset
The dataset used is EMNIST. You can find the data [here](https://www.kaggle.com/crawford/emnist/data). I have used the EMNIST gby merge dataset

## Requirements
1. Tensorflow
2. Keras
3. Numpy
4. h5py
5. Pandas
6. OpenCV

## Steps to execute
1. Keep the downloaded files in the same folder as the code. Execute the command "python convnet.py" to train the model.

2. The trained model is saved in the h5 file int the output folder.

3. For training a sample of own choice we can simply execute the command "python cannyedgedet.py lpp.jpg" which uses the image and segments the image. Each segmented image is tested using the test script which generates the output.

NOTE - The weights file is already present in the output folder. One can directly use the command "python cannyedgedet.py lpp.jpg" to generate output. First line of output is all the bounding box coordinates in the image and second line is the detected letters

## Results
Following are a few results obtained after training the model 

Canny edge detection - 
<img src="https://github.com/Shobhit20/MultiDetect-EMNIST/blob/master/Imgs/Canny.png" width="800">

Dilation
<img src="https://github.com/Shobhit20/MultiDetect-EMNIST/tree/master/Imgs/dilate.png" width="800">

Bounding box
<img src="https://github.com/Shobhit20/MultiDetect-EMNIST/blob/master/Imgs/BB.png" width="800">

Tested on multiple files
<img src="https://github.com/Shobhit20/MultiDetect-EMNIST/blob/master/Imgs/Result.png" width="400">

