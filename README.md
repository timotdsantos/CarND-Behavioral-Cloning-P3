# Behaviorial Cloning Project

[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

[//]: # (Image References)
[image1]: ./examples/NVIDIA_CNN.png "NVIDIA CNN Model Architecture" 
[image2]: ./examples/proc_shade.png "Processed Image to introduce difference in shade" 
[sample_c]: ./examples/sample_c.jpg "Center Image" 
[sample_l]: ./examples/sample_l.jpg "Left Image" 
[sample_r]: ./examples/sample_r.jpg "Right Image" 
[hist_all]: ./examples/hist_all.png "Histogram of Default Data Set" 
[hist_augmented]: ./examples/hist_augmented.png "Histogram of Final Training Set" 
[img_mse]: ./examples/mse.png "MSE of Training Epochs" 


Overview
---
This contains the discussion of my approach to the Behavioral Cloning Project.

In this project, I've learned about using deep neural networks and convolutional neural networks to clone driving behavior. Here, I trained, validated and tested models using Keras. The model's output is steering angle to an autonomous vehicle, based on images captured by the camera.

The provided simulator was used to train and test the model. You can steer a car around a track for data collection and the image data and steering angles were used to train a neural network. The model is integrated with the simulator so that the model can be used to drive the car autonomously around the track.

### Project Objectives
---

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior 
* Design, train and validate a model that predicts a steering angle from image data
* Use the model to drive the vehicle autonomously around the first track in the simulator. The vehicle should remain on the road for an entire loop around the track.


The Project
---

For the project, the following files are included in this repository: 
* [model.py](https://github.com/timotdsantos/CarND-Behavioral-Cloning-P3/blob/master/model.py)/[Behavioral_Cloning_Final.ipynb](https://github.com/timotdsantos/CarND-Behavioral-Cloning-P3/blob/master/Behavior_Cloning_Final.ipynb) contains the code to create and train the model. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.
* [model.h5](https://github.com/timotdsantos/CarND-Behavioral-Cloning-P3/blob/master/model.h5) is the trained Keras model

* [video.mp4](https://github.com/timotdsantos/CarND-Behavioral-Cloning-P3/blob/master/video.mp4) is a video recording of the vehicle driving autonomously around the track for at least one full lap
* [README.md](https://github.com/timotdsantos/CarND-Behavioral-Cloning-P3/blob/master/README.md) file is the overall writeup and discussion describing the steps taken to accomplish the project goals
* [drive.py](https://github.com/timotdsantos/CarND-Behavioral-Cloning-P3/blob/master/drive.py) (script to drive the car)

    Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```


### Dependencies
This project requires:

* [CarND Term1 Starter Kit](https://github.com/udacity/CarND-Term1-Starter-Kit)

The enviroment can be created with CarND Term1 Starter Kit. Click [here](https://github.com/udacity/CarND-Term1-Starter-Kit/blob/master/README.md) for the details.

The simulator can be downloaded from the classroom. 


## Model Architecture and Training Strategy

### 1. An appropriate model architecture has been employed

I started with just a single fully-connected layer in order to setup the framework for the iterations. Multiple models were implemented starting from simple architectures that were covered in the course materials (single layer, lenet) to more complex ones(NVIDIA CNN). 

In selecting which model architecture to use, I iterated using 1-3 epochs of training and compared the validation loss, wherein NVIDIA CNN was one of the well performing models.I noticed that there's improvement on the validation accuracy and increased training time as the complexity increased. In the final submission, I used the NVIDIA CNN.


### 2. Attempts to reduce overfitting in the model

The ideal number of epochs for the selected model (NVIDIA-CNN) was 6 as evidenced by the lack of significant improvement of the validation and training mean-square-error for epochs 7 and above, anything above would have been overfitted. 

![alt text][img_mse]


The final litmus test of the trained model is by running the simulator in autonomous mode using the trained model.


### 3. Model parameter tuning

In terms of optimizing the model parameters, I used an adam optimizer so that manually training the learning rate wasn't necessary. 

```
model.compile(loss='mse', optimizer='adam')```

