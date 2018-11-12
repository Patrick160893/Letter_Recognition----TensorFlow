#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 11 09:23:48 2018

@author: patrickorourke
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 31 09:37:08 2018

@author: patrickorourke
"""

# Assignment for the dataset "Auto MPG"

import numpy as np
import pandas as pd
import tensorflow as tf
import seaborn as sb
import matplotlib.pyplot as plt
from scipy.stats.stats import pearsonr   
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split



# Units - "Miles per galllon", "number", "Meters", "unit of power", "Newtons" . "Meters per sec sqr"
columns = ['mpg', 'cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'model year', 'origin', 'car name']

alpha = 'abcdefghijklmnopqrstuvwxyz'

learning_rate = 0.01
EPOCH = 100
# STEP 1 - GATHERING DATA

# Function to read textfile dataset and load it as a Pandas DataFrame
def loadData(file,columns):
    df = pd.read_table(file, sep=',')
    df.columns = columns
    return df

def correlation(data):
    correlation = []
    for i in range(0,7):
        j = pearsonr(data.iloc[:,i],data.iloc[:,0])
        correlation.append(j)
    return correlation

def charToOneHot(s):
        oneHot = [0 for _ in range(len(alpha))]
        oneHot[alpha.index(s.lower())] = 1
        return np.array(oneHot)
    
file = "/Users/patrickorourke/Documents/letter-recognition/letter-recognition.data.txt"
# Label the columsn of the Pandas DataFrame
columns = ['Letter', 'X-Box', 'Y-Box', 'Width', 'High', 'onpix', 'x-bar', 'y-bar', 'x2bar','y2bar','xybar','x2ybr','xy2br',
               'x-ege','xegvy','y-ege','yegvx']
data = loadData(file,columns)
    
# STEP 2 - PREPARING THE DATA
    
# Examine the dataset
data.head()
    
train, test = train_test_split(data, test_size=0.2)
    
ys_train = np.array(train.iloc[:,0].values)
    
ys_test = np.array(test.iloc[:,0].values)

train = train.iloc[:,1:17]
    
test = test.iloc[:,1:17]

inputs = tf.placeholder(tf.float32, shape = (1,16))
labels = tf.placeholder(tf.float32, shape = (1,26))
W_1 = tf.Variable(tf.random_uniform([16,50]))
W_2 = tf.Variable(tf.random_uniform([50,26]))
b_1 = tf.Variable(tf.zeros([50]))
b_2 = tf.Variable(tf.zeros([26]))
layer_1 = tf.add(tf.matmul(inputs,W_1), b_1)
layer_1 = tf.nn.sigmoid(layer_1)
layer_2 = tf.add(tf.matmul(layer_1,W_2), b_2)
predictions = tf.nn.softmax(layer_2)
# layer_2 = tf.nn.sigmoid(layer_2)

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = layer_2, labels = labels))

optim = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    train_losses = []
    for e in range(EPOCH):
        
        epoch_loss = 0
        for i in range(len(train)):
            inp = train.iloc[i, :].values / 15
            inp = inp.reshape(1, 16)
            lab = charToOneHot(ys_train[i]).reshape(1, 26)
            iter_loss, _ = sess.run([loss, optim], feed_dict = {inputs:inp, labels:lab})

            epoch_loss += iter_loss
        
        epoch_loss /= len(train)
        print(e, epoch_loss)
        train_losses.append(epoch_loss)
       
        
    test_loss = 0
    for i in range(len(test)):
        inp = test.iloc[i, :].values / 15
        inp = inp.reshape(1, 16)
        lab = charToOneHot(ys_test[i]).reshape(1, 26)
        iter_loss, _ = sess.run([loss, optim], feed_dict = {inputs:inp, labels:lab})
        test_loss += iter_loss
        
    test_loss /= len(test)
    
    print(test_loss)
    
    plt.plot(train_losses, label='train')
    plt.legend()  
