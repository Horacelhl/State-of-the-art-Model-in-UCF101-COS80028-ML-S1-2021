#################
# swinburne uni
# Creator: Qiyuan zhu
# Initialized: 28-03-2021
# Comments:
# This file contain teacher-student network all functions and models

import numpy as np
import pandas as pd
import tensorflow_hub as hub
from urllib import request
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input, Concatenate
from tensorflow.keras import layers

#  ===== Teacher network =====

# setting teacher model
i3d = hub.load("./i3d-kinetics-400_1.tar").signatures['default']

# setting label can be recognized
#KINETICS_URL = "./label_map.txt"
#with request.urlopen(KINETICS_URL) as obj:
#    labels = [line.decode("utf-8").strip() for line in obj.readlines()]
#    print("Found %d labels." % len(labels))
    
with open("./label_map.txt",'r') as obj:
    labels = [line.strip() for line in obj.readlines()]   
    print("Found %d labels." % len(labels))


def predict(sample_video):
    # Add a batch axis to the to the sample video.
    model_input = tf.constant(sample_video, dtype=tf.float32)[tf.newaxis, ...]

    logits = i3d(model_input)['default'][0]
    probabilities = tf.nn.softmax(logits)
    
    top_5_list = []
    print("Top 5 actions:")
    for i in np.argsort(probabilities)[::-1][:5]:
        top_5_list.append([labels[i],float(probabilities[i] * 100)])
        print(f"  {labels[i]:22}: {probabilities[i] * 100:5.2f}%")
    return top_5_list
        
# the size of clip is very important to think
# the large size of clip will make the final computation of probability too smooth, and probability concentrate on
# one of the region, which makes the selection will appear on region rather than single frame
# we might need to try 25%(4), 20%(5), 10%(10) and 5%(20) and also use the probability which can represent the distinguish ability
# rather than correct classification probability
def predict_all(whole_vedio, number_of_frame, clip_factor = 4):
    clip_size = number_of_frame//clip_factor   # we want whole integer (important part), default 25% in each clip window
    result = []
    for i in range(number_of_frame - clip_size + 1): # to achieve all clips
        print("block:", i)
        result.append(predict(tf.convert_to_tensor(whole_vedio[:, i:i+clip_size])[0]))
    return result

def search_target(target, myList):
    for i, ele in enumerate(myList):
        if target in ele:
            return ele[1]
        else:
            print("there is a nan in prob list")
            return 0
    return 0

# probability computation
# find which class holds highest pro
def find_highest_class(whole_result):
    flatten_result = [record for clip in whole_result for record in clip]
    df = pd.DataFrame(flatten_result,columns=["label","prob"])
    sum_df = df.groupby('label').sum().sort_values('prob',ascending=False)
    target_class = sum_df.reset_index().loc[0,'label']
    #print(sum_df)
    #print(sum_df.reset_index().loc[0,'label'])
    return target_class

# return: a list with len(whole result) with probability
def find_class_prob(target, whole_result):
    prob_list = []
    for each in whole_result:
        prob_list.append(search_target(target,each))

    return prob_list

def padding_based_on_pos(pos,prob,clip_len,number_frame):
    prob_seq = []
    
    for i in range(pos):   # padding at beginning
        prob_seq.append(0)
    
    for i in range(clip_len):  # record the probability
        prob_seq.append(prob)

    for i in range(number_frame - clip_len - pos):   # padding at end
        prob_seq.append(0)  
    
    return prob_seq

# based on highest prob class, select the prob from all result, and pad 0 when out range
def padding_prob(whole_result, number_frame, clip_factor):

    # first, extract the prob for target class
    target_class = find_highest_class(whole_result)
    prob_list = find_class_prob(target_class, whole_result)    
    #print(len(prob_list))
    
    # second create list and padding zero based on size of frame, and clip factor
    # the struture of data should look like below:
    #               frame    1         2         3         4         5         6  ............... n 
    # clip 1              prob(1)   prob(1)   prob(1)   prob(1)    prob(1)   prob(1)  .........
    #      2                        prob(2)   prob(2)   prob(2)    prob(2)   prob(2)  .........
    #      3                                  prob(3)   prob(3)    prob(3)   prob(3)  .........
    #      .  
    #      m  
    result_list = []
    clip_len = number_frame//clip_factor
    for i in range(len(prob_list)):
        result_list.append(padding_based_on_pos(i,prob_list[i],clip_len,number_frame))

    # transform list to dataframe?
    # print(len(result_list[0]))
    return result_list

# sum all prob and div the result by number -> the prob of importance we will use for individual frame
def compute_prob(prob_list, number_frame):
    frame_size = np.zeros(number_frame, np.float)
    count = np.zeros(number_frame, np.float)
    for each in prob_list:
        for i in range(len(each)):
            if each[i] != 0:
                count[i] = count[i] + 1
                frame_size[i] = frame_size[i] + each[i]
   
    final_prob = np.zeros(number_frame, np.float)
    for i in range(number_frame):
        if count[i] == 0:
            print("final prob found nan")
            final_prob[i] = 0
        else:
            final_prob[i] = frame_size[i] / count[i]

    return final_prob

def single_prob_generator(single_result, number_frame, clip_factor):
    prob_list = padding_prob(single_result, number_frame, clip_factor)
    final_prob = compute_prob(prob_list,number_frame)
    
    return final_prob

# To get all results
def prob_generator(vedio_list, number_of_frame, clip_factor = 4):
    whole_list_prob = []
    for i in range(len(vedio_list)):
        single_result = predict_all(vedio_list[i], number_of_frame, clip_factor)
        prob_list = padding_prob(single_result, number_of_frame, clip_factor)
        final_prob = compute_prob(prob_list,number_of_frame)
        whole_list_prob.append(final_prob)
    return whole_list_prob

#  ===== Student network =====
mlp_head_units = [256, 128] 

def mlp(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = layers.Dense(units, activation=tf.nn.gelu)(x)
        x = layers.Dropout(dropout_rate)(x)
    return x

def create_single_frame_selector():    
    
    input_shape = (1,3000)
    inputs = keras.layers.Input(shape=input_shape)

    representation = keras.layers.Dropout(0.5)(inputs)
    
    # Add MLP.
    #features = mlp(representation, hidden_units=mlp_head_units, dropout_rate=0.5)
    features = keras.layers.Dense(100)(representation)
    features = keras.layers.Flatten()(features)
    # Classify outputs.
    logits =  keras.layers.Dense(1)(features) #keras.activations.softmax(features,axis=-1)
    
    # Create the Keras model.
    model = keras.Model(inputs=inputs, outputs=logits)
    optimizer = tf.keras.optimizers.RMSprop(0.000005)
    model.compile(loss='mse', optimizer=optimizer, metrics=['mae'])
    return model