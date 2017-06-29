# Behaviorial Cloning Project

[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

Overview
---
This contains the discussion of my approach to the Behavioral Cloning Project.

In this project, I've learned about using deep neural networks and convolutional neural networks to clone driving behavior. Here, I trained, validated and tested models using Keras. The model's output is steering angle to an autonomous vehicle, based on images captured by the camera.

The provided simulator was used to train and test the model. You can steer a car around a track for data collection and the image data and steering angles were used to train a neural network and then use this model to drive the car autonomously around the track.

The Project
---

For the project, the following files are included in this repository: 
* model.py (script used to create and train the model)
* drive.py (script to drive the car)
* model.h5 (a trained Keras model)
* video.mp4 (a video recording of the vehicle driving autonomously around the track for at least one full lap)
This README file is the overall writeup and discussion describing the steps taken to accomplish the project goals.

Project Workflow
---


Project Goals
---
The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior 
* Design, train and validate a model that predicts a steering angle from image data
* Use the model to drive the vehicle autonomously around the first track in the simulator. The vehicle should remain on the road for an entire loop around the track.


### Dependencies
This project requires:

* [CarND Term1 Starter Kit](https://github.com/udacity/CarND-Term1-Starter-Kit)

The enviroment can be created with CarND Term1 Starter Kit. Click [here](https://github.com/udacity/CarND-Term1-Starter-Kit/blob/master/README.md) for the details.

The simulator can be downloaded from the classroom. 


## Behavioral Cloning Solution Design Approach

The project conssists of three sections:
- Data Collection and Augmentation
- Data Processing
- Model Development


### Data Collection and Augmentation

#### Default Data set
The data used was initially the one included in the project repository. In the initial iterations of the development, the included data was sufficient to be able to drive the car in the parts of the track with minimal curves.

Initially, there were around 8000+ images used, using only the 'center' camera of the car.

#### Recovery Data Set
Observing the behavior of the initial models, the car seemed to not be able to get back to the middle of the road when near the edge. To solve this, I recorded multiple occasions where the car is controlled to go back to the middle of the lane.

### Data Augmentation

#### Multiple-camera

#### Additional Data from Simulator

### Final Model Architecture

### Training and Validation

### Training and Validation


```sh
python video.py run1
```

The final model architecture (model.py lines 18-24) consisted of a convolution neural network based on the NVIDIA architecture.

Here is a visualization of the architecture.



#### Why create a video

1. It's been noted the simulator might perform differently based on the hardware. So if your model drives succesfully on your machine it might not on another machine (your reviewer). Saving a video is a solid backup in case this happens.
2. You could slightly alter the code in `drive.py` and/or `video.py` to create a video of what your model sees after the image is processed (may be helpful for debugging).
