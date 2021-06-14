#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.sequence import pad_sequences

import numpy as np
import pickle


# In[2]:


def generate_caption(photo, model):
    # initialize input text with start token
    input_text = 'start_'
    maxlen = 31
    
    for i in range(maxlen):
        seq = [word2idx[word] for word in input_text.split() if word in word2idx]
        # pad
        seq = pad_sequences([seq], maxlen=maxlen, padding='post')
        
        # predict
        y_pred = model.predict([photo, seq])
        y_pred = y_pred.argmax() # word with max probability
        predicted_word = idx2word[y_pred]
        # add predicted word to input text for next prediction
        input_text += ' ' + predicted_word
        
        if predicted_word == "end_":
            break
    caption = input_text.split()[1:-1]
    return " ".join([word for word in caption])
    


# In[3]:


# write image preprocessing function
def preprocess_img(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img = image.img_to_array(img)
    # convert to a batch
    img = np.expand_dims(img, axis=0)
    
    # Normalization
    img = keras.applications.resnet50.preprocess_input(img)
    
    return img

# write feature extraction function to encode images
def encode_img(img_path):
    # preprocess the image
    preprocessed_img = preprocess_img(img_path)
    # pass through the model and get features (1, 2048)
    feature_vector = feature_model.predict(preprocessed_img)
    # reshape feature vector to (2048,)
    feature_vector = feature_vector.reshape((-1,))
    return feature_vector


# In[4]:


# load trained model
cap_model = load_model('models/capit_model-20.h5')

# load resnet model
resnet = keras.applications.ResNet50()

# load word mapping dicts
with open("data/word2idx.pkl", "rb") as f:
    word2idx = pickle.load(f)
    
with open("data/idx2word.pkl", "rb") as f:
    idx2word = pickle.load(f)
    
# create feature_extractor model
feature_model = Model(inputs=[resnet.input], outputs=[resnet.layers[-2].output])


# In[5]:


def get_caption(image_path):
    img_featurevec = encode_img(image_path)
    img_featurevec = img_featurevec.reshape((1, 2048))
    caption = generate_caption(img_featurevec, cap_model)
    return caption

