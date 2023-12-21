#!/usr/bin/env python
# coding: utf-8

# # Main CNN model for bat call classification

# In[1]:


# Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import Callable
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPool2D, Dropout
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import load_model
from tensorflow_addons.metrics import F1Score
import math
import pickle
import cv2
import time
from sklearn.model_selection import train_test_split
import itertools_len as itertools
from itertools_len import product
import gc


# In[2]:


# memory optimization, see https://github.com/tensorflow/tensorflow/issues/31312#issuecomment-813944860
class ClearMemory(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        gc.collect()
        tf.keras.backend.clear_session()


# In[3]:


# class to track execution time of certain code events
class track_time:
    def __init__(self):
        self.events = []
        self.add('Start')
    def add(self, name: str) -> None:
        if name == "total":
            raise RuntimeError("Cant use the name 'total'.")
        self.events.append([name,time.time()])
    def get_time(self): # calculate time between events and total
        self.timed_events = {}
        for (n, event) in enumerate(self.events):
            elapsed_time = 0
            if n+1 == len(self.events):
                # last element
                elapsed_time = time.time() - event[1]
            else:
                elapsed_time = self.events[n+1][1] - event[1]
            self.timed_events[event[0]] = elapsed_time
        self.timed_events['total'] = time.time() - self.events[0][1]
        return self.timed_events
    def __str__(self):
        output = ""
        if not hasattr(self,'timed_events'):
            self.get_time()
        output += ("  Event tracked  |  Duration  \n")
        output += ("==============================\n")
        for name,duration in self.timed_events.items():
            output += (" "+name+"\t\t\t| "+str(round(duration,3))+"\n")
        return output


# In[4]:


# timer
timer = track_time()
timer.add("Read in data")
# load image data s and reshape 
data = pd.read_pickle('./data/images_df_numerical.pkl')
# convert to numpy array
X, y = data['data'], data['Species']
classes = y.unique()
image_size = X[0].size
samples = X.size
image_shape = (216,334,3) # height, width , channel
# reshape every row to the image, swap rgbs and scale to 0-1
X = [
    cv2.cvtColor(row.reshape(image_shape), cv2.COLOR_BGR2RGB).astype('float32')/255. 
    for row in X]
y = [row.astype('int32') for row in y]


# In[5]:


timer.add("Split Train/Test")
# Cross Valiadation, wenn wir ein 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1) 
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=1)

# conver to tensor for memory optimization
X_train = tf.convert_to_tensor(np.array(X_train))
y_train = tf.convert_to_tensor(np.array(y_train))

X_val = tf.convert_to_tensor(np.array(X_val))
y_val = tf.convert_to_tensor(np.array(y_val))

X_test = tf.convert_to_tensor(np.array(X_test))
y_test = tf.convert_to_tensor(np.array(y_test))


# In[6]:


# hyperparameter
number_of_classes = classes.size
pooling_size = (2, 2)
early_stopping = EarlyStopping(monitor='val_accuracy', patience=30, min_delta=0.001, start_from_epoch=15, restore_best_weights=True)
padding = "same"
epochs = 1
dropout_rate = 1 - 0.8 # ggf anpassen, wenn overfittet

def create_model(conv_kernel_sizes: list, conv_filter_nums: list, number_of_neurons: list, optimizer="adam", activation_function=lambda x: tf.math.maximum(x, 0.0)):
    f1 = F1Score(num_classes=number_of_classes, average="micro")

    model=Sequential()

    # adding activaation function seperate for memory optimization, 
    #   see https://github.com/tensorflow/tensorflow/issues/46475#issuecomment-817191096 and 
    #       https://github.com/tensorflow/tensorflow/issues/46475#issuecomment-1288677907
    
    model.add(Conv2D(conv_filter_nums[0], conv_kernel_sizes[0],activation=activation_function,input_shape=image_shape,padding=padding))
    #model.add(activation_function)
    # MaxPool2D((2, 2), strides=(2, 2), dtype="mixed_float16")(x)
    model.add(MaxPool2D(pooling_size, strides=(2, 2)))
    model.add(Dropout(dropout_rate))

    model.add(Conv2D(conv_filter_nums[1],conv_kernel_sizes[1],activation=activation_function, padding=padding))
    #model.add(activation_function)
    model.add(MaxPool2D(pooling_size, strides=(2, 2)))
    model.add(Dropout(dropout_rate))

    # Classficiation
    model.add(Flatten())
    model.add(Dense(number_of_neurons[0], activation=activation_function))
    #model.add(activation_function)
    model.add(Dropout(dropout_rate))

    model.add(Dense(number_of_neurons[1], activation=activation_function))
    #model.add(activation_function)
    model.add(Dropout(dropout_rate))

    model.add(Dense(number_of_neurons[2], activation=activation_function))
    #model.add(activation_function)
    model.add(Dropout(dropout_rate))

    # Output-Layer
    model.add(Dense(number_of_classes, activation="softmax"))
    model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=["accuracy", f1]) #,run_eagerly=True) # eagerly for memory optimization, see https://github.com/tensorflow/tensorflow/issues/31312#issuecomment-821809246

    tf.keras.backend.clear_session()
    gc.collect()
    return model
    


# In[7]:


print(y_test.shape)

batch_sizes = [8, 16, 32, 64, 128]
learning_rates = [0.01, 0.001, 0.0001, 0.00001, 0.000001]
conv_kernel_sizes = [[(3,3), (3, 3)],[(7,7), (3, 3)],[(11,11), (3, 3)]] # schauen, ob ggf. wir mehr layer benutzen
conv_filter_nums = [[32,32],[32, 64],[64,64],[64,32]]
number_of_neurons = [[256, 128, 64]]
histories_with_params = list()

test_list = product(batch_sizes, learning_rates, conv_kernel_sizes, conv_filter_nums, number_of_neurons)
print(f"Trying out {len(test_list)} different combination.")
# do again cause list
test_list = product(batch_sizes, learning_rates, conv_kernel_sizes, conv_filter_nums, number_of_neurons)

for batch_size, learning_rate, conv_kernel_size, conv_filter_num, number_of_neuron in test_list:    
    print(f"Now training model with bs={batch_size}, ls={learning_rate}, kn={conv_kernel_size[0]}x{conv_kernel_size[1]}, ft={conv_filter_num[0]}x{conv_filter_num[1]}, nn={number_of_neuron[0]}x{number_of_neuron[1]}x{number_of_neuron[2]}")
    model = create_model(conv_kernel_size, conv_filter_num,number_of_neuron)
    #model.summary()
    history = model.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        workers=8, # workers are number of cores
        callbacks=[early_stopping,ClearMemory()],
        validation_data=(X_val, y_val),
        verbose=1)
    
    #parameters = {
    #    "bs": batch_size, 
    #    "lr": learning_rate,
    #    "kn": conv_kernel_size,
    #    "ft": conv_filter_num,
    #    "nn": number_of_neuron,
    #    "ts": round(model.evaluate(X_test, y_test)[1], 2)*100
    #}
    
    #history_with_param = {"history": history, "parameters": parameters}
    #
    #histories_with_params.append(history_with_param)

    #model.save(f'./cnn_files/cnn_bs{batch_size}_ls{learning_rate}.keras')

    #print(f"Epochs: {len(history.history['accuracy'])}")
    #print(f"Test Score: {round(model.evaluate(X_test, y_test)[1], 2)}%")
    del model
    tf.keras.backend.clear_session()
    gc.collect()



# In[ ]:


#number_of_epochs = len(history.history["accuracy"])

#for history_with_param in histories_with_params:
    #model = load_model(f"./cnn_files/cnn_bs{
    #    history_with_param['parameters']['bs']
    #    }_ls{
    #    history_with_param['parameters']['lr']
    #    }_kn{
    #    history_with_param['parameters']['kn'][0]
    #    }x{
    #    history_with_param['parameters']['kn'][1]
    #    }_ft{
    #    history_with_param['parameters']['ft'][0]
    #    }x{
    #    history_with_param['parameters']['ft'][1]
    #    }_nn{
    #    history_with_param['parameters']['nn'][0]
    #    }x{
    #    history_with_param['parameters']['nn'][1]
    #    }x{
    #    history_with_param['parameters']['nn'][2]
    #    }.keras")
    
#    plt.plot(history_with_param["history"].history["val_accuracy"], label="val_data accuracy")
#    plt.plot(history_with_param["history"].history["accuracy"], label="train_data accuracy")
#    
#    plt.scatter(number_of_epochs, model.evaluate(X_test, y_test)[1], label="test_data accuracy", marker="x", c="g")
#    plt.title(f"bs{
#        history_with_param['parameters']['bs']
##        } ls{
#        history_with_param['parameters']['lr']
#        }\nkn{
#        history_with_param['parameters']['kn'][0]
#        }x{
#        history_with_param['parameters']['kn'][1]
#        } ft{
#        history_with_param['parameters']['ft'][0]
#        }x{
 #       history_with_param['parameters']['ft'][1]
 ##       }\nnn{
 #       history_with_param['parameters']['nn'][0]
 #       }x{
  #      history_with_param['parameters']['nn'][1]
#        }x{
#        history_with_param['parameters']['nn'][2]
#        }\nts{
#        history_with_param['parameters']['ts']
#        }")
#    plt.xlabel("Epochs")
#    plt.ylabel("Accuracy")
#    plt.legend(loc="lower right")
#    plt.savefig(f"./cnn_files/cnn_bs{
#        history_with_param['parameters']['bs']
#        }_ls{
#        history_with_param['parameters']['lr']
#        }_kn{
#        history_with_param['parameters']['kn'][0]
#        }x{
##        history_with_param['parameters']['kn'][1]
#        }_ft{
##        history_with_param['parameters']['ft'][0]
#        }x{
#        history_with_param['parameters']['ft'][1]
#        }_nn{
#        history_with_param['parameters']['nn'][0]
#        }x{
 #       history_with_param['parameters']['nn'][1]
 #       }x{
#       history_with_param['parameters']['nn'][2]
#        }.keras",dpi=200)
#    plt.show()



# In[ ]:


# prints n=|base_group| figures with subplots, based on the other paramters
#def print_results(base_group: (str,list), histories_all: list, number_of_epochs: int) -> None:
#    # for everx value of base group create figure, and than create subplots based on how many paramters there are
#    for val in base_group[1]:
#        # get all histires with said value
#        histories = [his for his in histories_all if his['parameters'][base_group[0]]==val]    
#        # Compute Rows required
#        total = len(base_group[1])
#        cols = int((total)**0.5)
#        rows = total // cols
#        if total % cols != 0:
#            rows += 1
#        pos = range(1,total+1)
#        
#        # plot
#        fig = plt.figure(figsize=(15,10))
#        for i in range(0,len(histories)):
#            # load model
#            model = load_model(f"./cnn_files/cnn_bs{histories[i]['parameters']['bs']}_ls{histories[i]['parameters']['lr']}.keras")
#            # get test score
#            test_score = round(model.evaluate(X_test, y_test)[1], 2)*100
#            # make a new subplot for every history
#            ax = fig.add_subplot(rows,cols,pos[i])
#            ax.set_ylim([0,1])
#            ax.set_xlim([0,number_of_epochs])
#            ax.plot(histories[i]["history"].history["val_accuracy"], label="val_data accuracy")
#            ax.plot(histories[i]["history"].history["accuracy"], label="train_data accuracy")
#            ax.set_title(f"{base_group[0]}: {histories[i]['parameters'][base_group[0]]} lr: {histories[i]['parameters']['lr']}, Test Score: {test_score}%")
#            ax.set_xlabel("Epochs")
#            ax.set_ylabel("Accuracy")
#            ax.legend(loc="lower right")
#        plt.savefig(f"t/{val}_a.png")        
#        plt.show()
#        plt.close()
#
#print_results(('bs',batch_sizes),histories_with_params,150)


# In[ ]:


#tf.keras.backend.clear_session()
#gc.collect()


# In[ ]:


# get memory usage

import sys

# These are the usual ipython objects, including this one you are creating
ipython_vars = ['In', 'Out', 'exit', 'quit', 'get_ipython', 'ipython_vars']

# Get a sorted list of the objects and their sizes
sorted_vars = sorted([(x, sys.getsizeof(globals().get(x))) for x in dir() if not x.startswith('_') and x not in sys.modules and x not in ipython_vars], key=lambda x: x[1], reverse=True)

sorted_vars_in_gb = [(var, size / (1024 ** 3)) for var, size in sorted_vars]
sorted_vars_in_gb
total_memory = sum(size for _, size in sorted_vars)
total_memory_in_gb = total_memory / (1024 ** 3)
total_memory_in_gb


# In[ ]:




