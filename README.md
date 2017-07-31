# Behaviorial Cloning Project

[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

[//]: # (Image References)

[image1]: ./examples/NVIDIA_CNN.png "NVIDIA CNN Model Architecture" 
[image2]: ./examples/recover_right.jpg "Recover Data from Right" 
[image3]: ./examples/proc_shade.png "Processed Data to introduce shadow" 
[sample_l]: ./examples/sample_l.jpg "Left Camera Image" 
[sample_r]: ./examples/sample_r.jpg "Right Camera Image" 
[sample_c]: ./examples/sample_c.jpg "Center Camera Image" 
Overview
---
This contains the discussion of my approach to the Behavioral Cloning Project.

In this project, I've applied deep neural networks and convolutional neural networks to clone driving behavior. Here, I trained, validated and tested models using Keras. The model's output is steering angle to an autonomous vehicle, based on images captured by the camera.

The provided simulator was used to train and test the model. You can steer a car around a track for data collection and the image data and steering angles were used to train a neural network and then use this model to drive the car autonomously around the track.

The Project
---

For the project, the following files are included in this repository: 
* model.py - this is the script used to create and train the model
* drive.py - the script to drive the car
* model.h5 - the a trained Keras model that works in conjunction with drive.py and the simulator that returns steering angles based on image recorded by the cameras

    ```python drive.py model.h5```
* run1_track1.mp4 - a video recording of the vehicle driving autonomously around track 1 for at least one full lap
This README file is the documentation and discussion describing the steps taken to accomplish the project goals.


Project Goals
---
The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior 
* Design, train and validate a model that predicts a steering angle from image data
* Use the model to drive the vehicle autonomously around the first track in the simulator. The vehicle should remain on the road for an entire loop around the track.

### Dependencies

* [CarND Term1 Starter Kit](https://github.com/udacity/CarND-Term1-Starter-Kit)

The enviroment can be created with CarND Term1 Starter Kit. Click [here](https://github.com/udacity/CarND-Term1-Starter-Kit/blob/master/README.md) for the details.

The simulator can be downloaded [here](https://github.com/udacity/self-driving-car-sim). 




## Behavioral Cloning Solution Design Approach

The project consists of three sections:
- Data Collection and Augmentation
- Data Processing
- Model Development


### Data Collection and Augmentation

#### Default Data set
The data used was initially the one included in the project repository. In the initial iterations of the development, the included data was sufficient to be able to drive the car in the parts of the track with minimal curves.

Initially, there were around 8000+ images used, using only the 'center' camera of the car. Other data augmentation approaches are discussed in the following sections.

#### Recovery Data Set
Observing the behavior of the initial models, the car seemed to not be able to get back to the middle of the road when near the edge. To solve this, I recorded multiple occasions where the car is controlled to go back to the middle of the lane. The steering angles are more aggressive towards the center in order to bring back the car to the middle. 

Here is an example of a recovery where the car is coming from the right lane.
![alt text][image2]



### Data Processing

#### Artificial Shadow Post-processing
Initially, sections of the track with various lighting and shading conditions became challenging spots in autonomous mode. My initial solution was to collect training data with similar shadow/shading conditions, but a more robust approach was chosen. From the training data, randomly selected images were post-processed to include different shadow condition. By making random portions of the image vary in shade, we are able to emulate the varying shadow condition that's usual in normal driving condition.

Here's an example of the post-processed image with the artificial shadow post-processing.
![alt text][image3]


#### Multiple-camera
The default data set and the simulator output contains three cameras (left, right, center). To be able to use the left and right cameras, the current steering angle is biased by 0.25 left or right when using the right and left images correspondingly. By doing this, we're practically getting 3 times the amount of data.

Here's the Left, Center, and Right camera images:

![alt text][sample_l]
![alt text][sample_c]
![alt text][sample_r]


#### Additional Data from Simulator
Another approach to add more data was to flip the images and multiply the corresponding steering angle by -1. The images where the steering angle is zero were not all selected and/or flipped.

#### Resampling
The distribution of angles were explored and I decided to randomly sample from the zero-angle data and from the +/- angles to dampen the effect of the overwhelming number of zero-angle training examples and get a more balanced distribution of steering angles.

### Final Model Architecture
The process of designing the architecture was iterative. I started implementing from basic network and built up to more complex networks. As the network became more complex, the training time per epoch increases, but it gained validation accuracy improvement.
The Final model selected was patterned after the NVIDIA architecture shown below.
 
![alt text][image1]

### Training and Validation



```sh
python video.py run1
```

The final model architecture (model.py lines 18-24) consisted of a convolution neural network based on the NVIDIA architecture.

Here is a visualization of the architecture.
![alt text][image3]


#### Why create a video

1. It's been noted the simulator might perform differently based on the hardware. So if your model drives succesfully on your machine it might not on another machine (your reviewer). Saving a video is a solid backup in case this happens.
2. You could slightly alter the code in `drive.py` and/or `video.py` to create a video of what your model sees after the image is processed (may be helpful for debugging).
