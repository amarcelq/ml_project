#!/usr/bin/env python
# coding: utf-8

# Nur nen test zu autoencodern

# In[2]:




# In[3]:


import tensorflow.keras as keras
from keras import layers
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


# In[41]:


# load images + class
images = pd.read_pickle('images_flatten.pkl').to_numpy()
image_size = images[0][1].size
image_shape = (216,334,3) # height, width , channels
x_train,x_test = train_test_split(images[:, 1:],test_size=0.2) 
x_train = np.vstack(x_train[:, 0]).astype('float32') / 255. # normalized, cause just better
x_test = np.vstack(x_test[:, 0]).astype('float32') / 255.

all_data = np.vstack(images[:, 1:][:, 0]).astype('float32') / 255.


# In[44]:


# very very simple 1 layer autoencoder
#encoded_dimension = 10

#input_img = keras.Input(shape=(image_size,))
#encoded = layers.Dense(encoded_dimension,activation='relu')(input_img)
#decoded = layers.Dense(image_size,activation='sigmoid')(encoded)

# model that maps input to guessed output
#autoencoder = keras.Model(input_img,decoded)

# model that maps input to the "compressed" representation
#encoder = keras.Model(input_img,encoded)

# model that maps representation to output
#encoded_input = keras.Input(shape=(encoded_dimension,))
#decoded_output = autoencoder.layers[-1](encoded_input) # same as layers.Dense(image_size,activation='sigmoid')(encoded_input) but retrived from other model
#decoder = keras.Model(encoded_input,decoded_output)

# cnn as en/decoder
shapes = image_shape

inputs = keras.Input(shape=shapes)
encoded = layers.Conv2D(32, (7, 7), activation='relu', padding='same')(inputs)
encoded = layers.MaxPooling2D((2, 2), padding='same')(encoded)
encoded = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
encoded = layers.MaxPooling2D((2, 2), padding='same')(encoded)

# Decoder
decoded = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(encoded)
decoded = layers.UpSampling2D((2, 2))(decoded)
decoded = layers.Conv2D(32, (7, 7), activation='relu', padding='same')(encoded)
decoded = layers.UpSampling2D((2, 2))(decoded)
output = layers.Conv2D(shapes[2], (3, 3), activation='sigmoid', padding='same')(decoded)

autoencoder = keras.Model(inputs,output)
# compile autoencoder
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

# reshape into image shape
stacked_train = x_train.reshape((x_train.shape[0],image_shape[0],image_shape[1],image_shape[2]))
stacked_test = x_test.reshape((x_test.shape[0],image_shape[0],image_shape[1],image_shape[2]))

autoencoder.summary()

# train on data
autoencoder.fit(stacked_train,stacked_train,
                epochs=10,
                batch_size=10,
                shuffle=True,
                validation_split=0.2
               )

# predict some images
#encoded_imgs = encoder.predict(x_test)
#decoded_imgs = decoder.predict(encoded_imgs)
decoded_imgs = autoencoder.predict(stacked_test)
print(decoded_imgs.size)


# In[7]:


#print(decoded_imgs)
autoencoder.save('test_autoencoder_cnn.keras')


# In[45]:


# display some of the images vs reconstruction
n = 20  # How many digits we will display
plt.figure(figsize=(20, 4))
for i in range(n):
    # Display original
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(stacked_test[i].reshape(image_shape))
    #plt.color()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # Display reconstruction
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_imgs[i].reshape(image_shape))
    #plt.color()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.savefig("test_autoencoder_cnn_reconstruction.png",dpi=300)


# In[ ]:




