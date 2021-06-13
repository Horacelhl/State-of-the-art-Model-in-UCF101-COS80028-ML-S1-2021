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


mlp_head_units = [512,128]

def mlp(x, hidden_units, dropout_rate):
    print(hidden_units)
    for units in hidden_units:
        print(units)
        x = layers.Dense(units, activation=tf.nn.gelu)(x)
        x = layers.Dropout(dropout_rate)(x)
    return x


def create_global_frame_selector(num_class):
    #input_shape = (1,129440)
    input_shape = (1,6000)
    # inputs is Z1
    inputs = layers.Input(shape=input_shape)
    inputs2 = layers.Flatten()(inputs)
    Z1 = layers.Dense(512,activation='relu')(inputs2)
    
    # get alpha_i and Z2
    alpha_i = layers.Dense(1024, activation="sigmoid",name="alpha_i")(Z1)
    Z2 = tf.math.divide(tf.math.multiply(alpha_i,Z1),alpha_i)
    Z_concat = Concatenate(axis=1)([Z1,Z2])

    
    # get beta_i and omega_i
    beta_i = layers.Dense(512, activation="sigmoid",name="beta_i")(Z_concat)
    omega_i = tf.math.multiply(beta_i,Z1)
    omega_i_exp = tf.expand_dims(omega_i, axis=-2)    
    print("omega",omega_i.shape)
    
    # get h_i and lambda_i
    h_i = layers.LSTM(512)(omega_i_exp)
    print("hi",h_i.shape)
    #h_i = tf.expand_dims(h_i, axis=-2)
    lambda_i = layers.Softmax()(h_i)
    print("lambda",lambda_i.shape)
    Z3 = tf.math.divide(tf.math.multiply(lambda_i,omega_i),lambda_i)  
    
    # get global prob Gamma_i
    final_concat = Concatenate(axis=1)([omega_i,Z3])
    Gamma_i = layers.Dense(512,  activation="sigmoid",name="Gamma_i")(final_concat)
    print("gama",Gamma_i.shape)
    # following is for prediction
    # Add c_t.
    # Add MLP.
    # Classify outputs.
    # in a real usage we will drop these tail layer
    c_t =  tf.math.multiply(Gamma_i,h_i)
    print("ct",c_t.shape)
    representation = layers.Flatten()(c_t)
    #features = mlp(representation, hidden_units=mlp_head_units, dropout_rate=0.5)

    print("representation",representation.shape)
    features = layers.Dense(128)(representation)
    logits = layers.Dense(num_class)(features)
    
    # Create the Keras model.
    model = keras.Model(inputs=inputs,  outputs=logits)
    optimizer = tf.keras.optimizers.RMSprop(0.001)
    model.compile(loss=keras.losses.CategoricalCrossentropy(from_logits=True,reduction=tf.keras.losses.Reduction.NONE),
                  optimizer=optimizer, 
                  metrics=[tf.keras.metrics.CategoricalAccuracy(name="accuracy")])


    return model