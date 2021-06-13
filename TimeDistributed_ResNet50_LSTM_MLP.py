#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import time
import math
import numpy as np
from tqdm import tqdm
import pandas as pd
from glob import glob
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras import Model, Input, optimizers
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.layers import Dense, Dropout, TimeDistributed, LSTM, Flatten
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, multilabel_confusion_matrix, accuracy_score, ConfusionMatrixDisplay, classification_report

strategy = tf.distribute.MirroredStrategy()
#mirrored_strategy = tf.distribute.MirroredStrategy(devices=["/gpu:0", "/gpu:1"])
print ("#" * 100)
print ('Number of devices: {}'.format(strategy.num_replicas_in_sync))
#print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
print ("#" * 100) 


# In[2]:


VERSION = 2000
EPOCHS = 300
BATCH_SIZE = 16
input_size = 2, 224, 224, 3
sec_path = './training_result/'


# In[3]:


def get_frames(images):
    frames=[]
    for i in np.arange(len(images)):
        vid_name = images[i].split('.')[0].split('frame')[0]
        frames_to_select=[]
        for l in np.arange(0, 2):
            frames_to_select.append('frame%d.jpeg' % l)
        vid_data=[]
        for frame in frames_to_select:                
            frame_image = image.load_img(vid_name + frame, target_size=(224, 224))
            frame_image = image.img_to_array(frame_image)
            frame_image = preprocess_input(frame_image)        
            datu=np.asarray(frame_image)
            normu_dat=datu/255
            vid_data.append(normu_dat)
        vid_data=np.array(vid_data)
        frames.append(vid_data)    
    return np.array(frames)


# In[4]:


# USE THIS
images = sorted(glob("full_dataset/*.jpeg"))

train_image = []
train_class = []
for i in range(len(images)):    
    train_image.append(images[i].split('\\')[-1].split('/')[-1])    
    train_class.append(images[i].split('\\')[-1].split('_')[-4].split('/')[-1])
    
# storing the images and their class in a dataframe
train_data = pd.DataFrame()
train_data['image'] = train_image
train_data['class'] = train_class

# save csv file for future use 
train_data.to_csv('UCF_10.csv',header=True, index=False)

X = get_frames(images)
y = pd.get_dummies(train_data['class'])


# In[5]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)



# In[6]:


def plot_history(history):
    loss_list = [s for s in history.history.keys() if 'loss' in s and 'val' not in s]
    val_loss_list = [s for s in history.history.keys() if 'loss' in s and 'val' in s]
    acc_list = [s for s in history.history.keys() if 'acc' in s and 'val' not in s]
    val_acc_list = [s for s in history.history.keys() if 'acc' in s and 'val' in s]    
    if len(loss_list) == 0:
        print('Loss is missing in history')
        return     
    ## As loss always exists
    epochs = range(1,len(history.history[loss_list[0]]) + 1)
    
    fig = plt.figure(figsize=(16,10))
    
    ## Loss
    fig.add_subplot(2,1,1)
    for l in loss_list:
        plt.plot(epochs, history.history[l], 'b', label='Training loss (' + str(str(format(history.history[l][-1],'.5f'))+')'))
    for l in val_loss_list:
        plt.plot(epochs, history.history[l], 'g', label='Validation loss (' + str(str(format(history.history[l][-1],'.5f'))+')'))
    
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    ## Accuracy
    fig.add_subplot(2,1,2)
    for l in acc_list:
        plt.plot(epochs, history.history[l], 'b', label='Training accuracy (' + str(format(history.history[l][-1],'.5f'))+')')
    for l in val_acc_list:    
        plt.plot(epochs, history.history[l], 'g', label='Validation accuracy (' + str(format(history.history[l][-1],'.5f'))+')')

    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(sec_path + 'training_history_LSTM_v%d.png'%VERSION)
    plt.show()    


# In[7]:


def pre_trained_ResNet50():    
    resnet_weights_path = './resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'
    resnet_model = ResNet50(weights=resnet_weights_path, include_top=False, input_shape=(224, 224, 3))
    for layer in resnet_model.layers:
        layer.trainable = False
    return resnet_model


# In[12]:


def get_compiled_model():  
    ResNet50 = pre_trained_ResNet50()    
    intermediate_model= Model(inputs=ResNet50.input, outputs=ResNet50.get_layer('conv5_block3_out').output)
    input_tensor = Input(shape=input_size)
    timeDistributed_layer = TimeDistributed(intermediate_model)(input_tensor)
    timeDistributed_layer = TimeDistributed(Flatten())(timeDistributed_layer)
    lstm_layer = LSTM(512,return_sequences=False,dropout=0.2)(timeDistributed_layer)    
   
    MLP_layer = Dense(512)(timeDistributed_layer)
    MLP_layer = Dense(256)(MLP_layer)   
    MLP_layer = Dense(128)(MLP_layer) 
    MLP_layer = Dense(64)(MLP_layer)
    Flatten_layer = Flatten()(MLP_layer)
    output_layer = Dense(y.shape[1], activation='softmax')(Flatten_layer)    
    final_model = Model(inputs = input_tensor, outputs = output_layer)
    print(final_model.summary())
    
    return final_model


# In[ ]:


with strategy.scope():    
    
    final_model = get_compiled_model()
    final_model.compile(optimizers.Adam(lr=0.0001),
                  loss='categorical_crossentropy', metrics=['accuracy'])
    
    model_checkpoint_callback = ModelCheckpoint(sec_path + 'weight_LSTM_v%d.hdf5'%VERSION, save_best_only=True, monitor='val_loss', mode='min')
    es_callback = EarlyStopping(monitor='loss', patience=3)

# fit model
    start_time = time.time()
    history = final_model.fit(X_train, y_train, epochs=EPOCHS, 
                        validation_data=(X_test, y_test), batch_size=BATCH_SIZE, callbacks=[model_checkpoint_callback, es_callback])
    
    elapsed_time = time.time() - start_time # training time
    
# evaluate model
    loss, accuracy = final_model.evaluate(X_test, y_test)
    rounded_predictions = final_model.predict(X_test, batch_size=BATCH_SIZE, verbose=0)
# learning curves    
    plot_history(history)
    loss = float("{0:.3f}".format(loss))
    accuracy = float("{0:.3f}".format(accuracy))
    elapsed_time = float("{0:.3f}".format(elapsed_time))

#saving model
    hist_df = pd.DataFrame(history.history) 
    result = ["Evaluation result: ","loss: "+ str(loss), "accuracy: "+ str(accuracy), "elapsed_time: "+ str(elapsed_time)]
    result = pd.DataFrame(result)
    with open(sec_path + 'history_LSTM_v%d.csv'%VERSION, mode='w') as f:
        result.to_csv(f)
        hist_df.to_csv(f)


# In[ ]:
labels = np.unique(train_class)

#as_dict={}
cr_dict={}

f, axes = plt.subplots(2, 5, figsize=(25, 15))
axes = axes.ravel()
for i in range(len(labels)):
    y_true_label = np.asarray(y_test)[:, i]
    y_pred_label = rounded_predictions[:, i].round()
    #as_dict[labels[i]] = accuracy_score(y_pred=y_pred_label, y_true=y_true_label)    
    cr_dict[labels[i]] = classification_report(y_pred=y_pred_label, y_true=y_true_label)
    disp = ConfusionMatrixDisplay(confusion_matrix(y_pred=y_pred_label, y_true=y_true_label))
    disp.plot(ax=axes[i])
    disp.ax_.set_title(f'class {i}')
    disp.im_.colorbar.remove()

plt.subplots_adjust(wspace=0.10, hspace=0.1)
f.colorbar(disp.im_, ax=axes)
plt.savefig(sec_path + 'ConfusionMatrixDisplay_LSTM_v%d.png'%VERSION)
plt.show()   


for label, report in cr_dict.items():
    print("{}:".format(label))
    print(report)


