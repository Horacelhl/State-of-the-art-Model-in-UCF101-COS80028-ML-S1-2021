#################
# swinburne uni
# Creator: Qiyuan zhu
# Initialized: 28-03-2021
# Comments:
# This file contain all the data reading, and feature processing functions


# part 1
import sys
import imageio
import numpy as np
import skimage.transform
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import pandas as pd
from urllib import request  # requires python3
from scipy import spatial
from sklearn.manifold import TSNE
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

# part 2
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import MobileNetV3Large
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Concatenate
from tensorflow.keras import layers
import random

# part 3
from tensorflow import keras

# Video options
_INPUT_SIZE = 224
_NUM_FRAMES = 64

# ======= data reading =======
# we read all vedio and store into the list
def load_video(fn):

    video = np.ndarray((1, _NUM_FRAMES, _INPUT_SIZE, _INPUT_SIZE, 3), np.float32)
    reader = imageio.get_reader(fn)

    for i, im in zip(range(_NUM_FRAMES), reader):
        # Convert to float
        im = im / 255
        # Scale
        h, w = im.shape[:2]
        min_side = min(h, w)
        scale_factor = _INPUT_SIZE/min_side
        im = skimage.transform.resize(im, (int(h*scale_factor), int(w*scale_factor)))
        # Center crop
        h, w = im.shape[:2]
        im = im[(h-_INPUT_SIZE)//2:(h+_INPUT_SIZE)//2,
                (w-_INPUT_SIZE)//2:(w+_INPUT_SIZE)//2]
        video[:, i] = im
        
    return video
    
def show_video(video):
    plt.subplot(1, 3, 1)
    plt.imshow(video[:, _NUM_FRAMES//4].squeeze())
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(video[:, _NUM_FRAMES//2].squeeze())
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(video[:, 3*_NUM_FRAMES//4].squeeze())
    plt.axis('off')
    plt.show()
    
def load_dataset(base_path,set_length = None):
    video_list = []
    label = []
    count = 0   
    # for testing purpose, we only take first 100 video, almost 4GB in memory
    # 4GB still too much for my PC, I decline to 50 sample video
    
    if set_length is not None:
        for folder in os.listdir(base_path):
            for video in os.listdir(base_path+"\\"+folder):
                if video.endswith('.avi'):
                    test_vedio = load_video(base_path + "\\" + folder + "\\" + video)
                    #print(folder)
                    #show_video(test_vedio)
                    label.append(folder)
                    video_list.append(test_vedio)
                    count += 1
                
                # if not stop length we read all data, otherwise stop
                if count >= set_length:    
                    break
            if count >= set_length:
                break

    else:
        for folder in os.listdir(base_path):
            for video in os.listdir(base_path+"\\"+folder):
                if video.endswith('.avi'):
                    test_vedio = load_video(base_path + "\\" + folder + "\\" + video)
                    #print(folder)
                    #show_video(test_vedio)
                    label.append(folder)
                    video_list.append(test_vedio)
                    count += 1

    return video_list, label
    
def get_list_label(vedio_label):
    output_label = le.fit_transform(vedio_label)

    return output_label
    
# ======= data processing =======
# load vedio frame into a list
def data_load(video_list, label_list):
    data_input = []
    data_label = []
    for i in range(len(video_list)):
        for j in range(len(video_list[i][0])):
            data_input.append(video_list[i][:,j])
            data_label.append(label_list[i])
    data_input = np.array(data_input)
    data_label = np.array(data_label)
    #data_input = np.squeeze(data_input)
    print(data_input.shape)
    return data_input,data_label

# visual encoding models
input_shape = (224,224,3)
model1 = MobileNetV3Large(weights='imagenet', input_shape = input_shape, include_top=False)

# word encoding models
model2 = MobileNetV3Large(weights='imagenet', input_shape = input_shape, include_top=True)
embeddings_dict = {} # Read the pretrained weights

with open("glove.6B/glove.6B.200d.txt", 'r', encoding="utf-8") as f:
    for line in f:
        values = line.split()
        word = values[0]
        vector = np.asarray(values[1:], "float32")
        embeddings_dict[word] = vector

def select_word():
    result_label = []
    for item in pred_result:
        for ele in item:
            print(ele[1])
            result_label.append(ele[1])
    return result_label

def find_closest_embeddings(embedding):
    return sorted(embeddings_dict.keys(), key=lambda word: spatial.distance.euclidean(embeddings_dict[word], embedding))

def get_embeddings(embedding, result_label):
    result_embeddings = []
    for each in result_label:
        if each in  embedding:
            result_embeddings.append(embeddings_dict[word])
            #print(each, "each")
        else:
            result_embeddings.append(np.zeros(200))  # we use 200 dim Glove pretrained weights
    return result_embeddings

def label_processing(mobilenet_class_label,embedding):
    pred_result = tf.keras.applications.mobilenet_v3.decode_predictions(mobilenet_class_label, top=10)
    result_label = []
    for item in pred_result:
        for ele in item:
            result_label.append(ele[1])
    result = get_embeddings(embedding, result_label)
    result = np.array(result).reshape(1,-1)
    return result

# visual + word embedding processing
def feature_extraction(data_input):
    
    prepared_input = []
    for i in range(len(data_input)):
        
        #with tf.device("cpu:0"): prediction = model.predict()
        #with tf.device("cpu:0"): model1_out = model1.predict(data_input[i])
        model1_out = model1.predict(data_input[i])  # (None, 7, 7, 1280)
        model1_out = layers.Flatten()(model1_out)
        #print(model1_out.shape, "result_1")
        model1_out = layers.Dense(1000, activation='softmax')(model1_out) 
        #print(model1_out.shape, "result_model1_out")

        model2_out = model2.predict(data_input[i])  
        model2_out = label_processing(model2_out,embeddings_dict) # (None, 10, 200)
        model2_out = layers.Flatten()(model2_out)
        #print(model2_out.shape, "result_model2_out")
        
        concatenated = Concatenate(axis=1)([model1_out,model2_out])
        prepared_input.append(concatenated)
        #print(concatenated.shape, "concatenated")
        
    return prepared_input
    
# random pair generation
def random_pair_generation(prepared_input):
    prepared_random_pair = []
    for i in range(len(prepared_input)):
        Z1 = prepared_input[i]
        random_int = random.randint(0,len(prepared_input)-1)
        Z2 = prepared_input[random_int]
        temp_Z_pair = Concatenate(axis=1)([Z1,Z2])
        prepared_random_pair.append(temp_Z_pair)

    prepared_pair = tf.convert_to_tensor(prepared_random_pair)
    print(prepared_pair.shape, "prepared_pair")

    return prepared_pair
