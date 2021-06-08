#!/usr/bin/env python
# coding: utf-8

# In[7]:


import math 
import os
import numpy as np    
from tqdm import tqdm
import pandas as pd
import cv2


# In[8]:


path = "UCF_test/"
dataset = []
for path, subdirs, files in os.walk(path):
    for name in files:
        dataset.append(os.path.join(path, name))


# In[9]:


for i in tqdm(np.arange(len(dataset))):    
    video_name=dataset[i]
    video_read_path=video_name
    cap = cv2.VideoCapture(video_read_path)
    frameRate = cap.get(5) #frame rate
    x=1
    count = 0
    while(cap.isOpened()):
        frameId = cap.get(1) #current frame number
        ret, frame = cap.read()
        if (ret != True):
            break
        if (frameId % math.floor(frameRate) == 0):
            filename = 'train_new1/' + video_name.split('.')[0].split('_', 2)[-1] + "_frame%d.jpeg" % count;count+=1            
            cv2.imwrite(filename, frame)      
    cap.release()

