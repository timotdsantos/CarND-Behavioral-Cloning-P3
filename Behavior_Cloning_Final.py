
# coding: utf-8

# In[5]:

import csv
import cv2
import numpy as np
import random 
import pandas as pd
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

lines=[]

with open('im/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)
def process_shade(image):
    h, w = image.shape[0], image.shape[1]
    [x1, x2] = np.random.choice(w, 2, replace=True)
    k = h / (0.0001+x2 - x1)
    b = - k * x1
    for i in range(h):
        c = int((i - b) / (k+0.0001))
        image[i, :c, :] = (image[i, :c, :] * .65).astype(np.int32)
    return image        


# In[6]:

images = []
measurements = []


# In[7]:

log_pd = pd.read_csv('im/driving_log.csv')


# In[8]:

measurements = log_pd['steering']
image_fname = log_pd['center']
image_rname = log_pd['right']
image_lname = log_pd['left']


# In[9]:

print(measurements.shape)
print(image_fname.shape)
print(measurements[0:2])
print(image_fname[0:2])
print(type(measurements))
print(type(image_fname))
measurements=np.array(measurements)
print(type(measurements))


# In[10]:

y=[]

for i in range(len(image_fname)):
    if((measurements[i]==0)&(random.random()<0.8)):
        continue
    else:
        if(measurements[i]!=0):
            image = cv2.imread('im/' + image_rname[i].lstrip())
            if(random.random()>0.9):
                image=process_shade(image)
            images.append(image)
            y.append(measurements[i]-0.25)       
            image = cv2.imread('im/' + image_lname[i].lstrip())
            if(random.random()>0.9):
                image=process_shade(image)
            images.append(image)
            y.append(measurements[i]+0.25)   
        image = cv2.imread('im/' + image_fname[i].lstrip())
        if(random.random()>0.9):
            image=process_shade(image)
        images.append(image)
        y.append(measurements[i])            


# In[11]:

temp_x = np.array(images)
temp_y = np.array(y)


# In[12]:

print(temp_x.shape)
print(temp_y.shape)


# In[13]:

X_train = np.append(temp_x,np.fliplr(temp_x),axis=0)
y_train = np.append(temp_y,-1.0*temp_y,axis=0)


# In[14]:

print(X_train.shape)
print(y_train.shape)


# In[15]:

from keras.models import Sequential, Model
from keras.layers import Flatten, Dense
from keras.layers import Convolution2D, Cropping2D, MaxPooling2D
from keras.layers import Lambda

model = Sequential()

model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70,25),(0,0))))
model.add(Convolution2D(24, 5, 5, activation = 'relu', subsample=(2, 2) ))

model.add(Convolution2D(36, 5, 5, activation = 'relu', subsample=(2, 2) ))

model.add(Convolution2D(48, 5, 5, activation = 'relu', subsample=(2, 2) ))

model.add(Convolution2D(64, 3, 3, activation = 'relu' ))
# model.add(MaxPooling2D())

model.add(Convolution2D(64, 3, 3))
# model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool'))

# model.add(MaxPooling2D())
# model.add(Dropout(0.5))

model.add(Flatten())
model.add(Dense(1162))
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')

model.fit(X_train,y_train, validation_split=0.2,shuffle=True, nb_epoch=1,batch_size=256)

model.save('model_temp.h5')


# In[17]:

from keras.models import load_model

model = load_model('model_nvidia_5ep_postproc.h5')

model.fit(X_train,y_train, validation_split=0.2,shuffle=True, nb_epoch=1,batch_size=256)
model.save('model_temp_cont.h5')
    


# In[18]:

model = load_model('model_temp_cont.h5')

model.fit(X_train,y_train, validation_split=0.2,shuffle=True, nb_epoch=5,batch_size=256)
model.save('model_temp_cont_6ep.h5')
    


# In[ ]:



