#!/usr/bin/env python
# coding: utf-8

'''
у нас есть код:
***
clear all;
close all;
clc;

%% Load Data
load('Data')
load('Primary_data')

D=D_Max(~strcmp(D_Max,'None'));
X=Data(:,:,~strcmp(D_Max,'None'));

D=reshape(str2double(D),1,[]);


%% Initialize parameters
CC=[2 2 20];                    % Cross correlation architecture 4*3*20
hidden_layers=[10 10 10];        % 3 layers
epoch=100;

tic
[WC_Dmax, net_Dmax, tr_Dmax]=trainConv(X(:,:,2:end),D(1,2:end),hidden_layers, CC, epoch);
toc


    for k = 1:length(D)
        %% Data Correlating
        x    = X(:, :, k);
        yC1  = Conv(x, WC_Dmax);
        yC2  = ReLU(yC1);
        yC   = Pool(yC2);
        %% Data Flattening
        yC_f         = reshape(yC, [], 1);
        x_f          = reshape(x, [], 1);
        x_flattened(:,k)  = [yC_f;x_f];
    end

t=D;
y=net_Dmax(x_flattened);
plotregression(t,y)
performance = perform(net_Dmax,t,y)

load gong.mat;
sound(y);


save('CBNN_Dmax.mat','WC_Dmax','net_Dmax','tr_Dmax');

***
'''
# Мне непонятна вот эта часть:
# ![%D0%A1%D0%BD%D0%B8%D0%BC%D0%BE%D0%BA%20%D1%8D%D0%BA%D1%80%D0%B0%D0%BD%D0%B0%202023-11-27%20%D0%B2%2013.22.21.png](attachment:%D0%A1%D0%BD%D0%B8%D0%BC%D0%BE%D0%BA%20%D1%8D%D0%BA%D1%80%D0%B0%D0%BD%D0%B0%202023-11-27%20%D0%B2%2013.22.21.png)
#
# Ее можно реализовать как:
# python
# import numpy as np
# import torch
#
#
# tic = time.time()
#
# X_slice = X[:, :, 1:]
# D_slice = D[0, 1:]
#
# WC_Dmax, net_Dmax, tr_Dmax = trainConv(torch.Tensor(X_slice), torch.Tensor(D_slice), hidden_layers, CC, epoch)
#
# toc = time.time()
#
# Которая выполняет такие действия:
#  использует библиотеки numpy и torch для манипуляции данными и обучения сверточной нейронной сети. Он выполняет следующие действия:
#
# 1. Импортирует библиотеки numpy и torch.
# 2. Предполагается, что переменная X - это трехмерный массив numpy, а D - двумерный массив numpy.
# 3. Выполняет нарезку данных: X_slice получается выбором всех элементов вдоль первой и второй осей массива X, и всех элементов начиная с индекса 1 вдоль третьей оси. D_slice получается выбором первого элемента вдоль первой оси массива D и всех элементов начиная с индекса 1 вдоль второй оси.
# 4. Запускает функцию trainConv с использованием полученных на предыдущем шаге данных X_slice и D_slice, а также других параметров hidden_layers, CC, epoch. Результаты этой функции записываются в переменные WC_Dmax, net_Dmax и tr_Dmax.
# 5. Замеряет время выполнения кода с помощью tic и toc.
#
# Функция trainConv, видимо, реализована на PyTorch и используется для обучения сверточной нейронной сети на входных данных X_slice и D_slice.

# In[2]:


# ### Вариант 1

# In[1]:

import numpy as np
import tensorflow as tf

from Conv import Conv
from Pool import Pool
from ReLU import ReLU

# Clear all variables
tf.resetdefaultgraph()

# Load Data
Data = np.load('Data.npy')
Primary_data = np.load('Primary_data.npy')

D_Max = np.load('D_Max.npy')
D = D_Max[np.where(D_Max != 'None')].astype(np.float32)
X = Data:,:, np.where(D_Max != 'None')[0]

# D_Max = D_Max[D_Max != 'None']
# X = Data[:, :, D_Max != 'None']
# D = np.array(list(map(float, D_Max)))


# Initialize parameters
CC = 2, 2, 20  # Cross correlation architecture 4*3*20
hidden_layers = [10, 10, 10]  # 3 layers
epoch = 100


# Define the model
def model(X, D, hidden_layers, CC, epoch):
    WC_Dmax, net_Dmax, tr_Dmax = None, None, None

    # Training Convolutional Neural Network
    with tf.Session() as sess:
        sess.run(tf.globalvariablesinitializer())

        for e in range(epoch):
            for k in range(len(D)):
                # Data Correlating
                x = X[:, :, k]
                yC1 = Conv(x, WC_Dmax)
                yC2 = ReLU(yC1)
                yC = Pool(yC2)

                # Data Flattening
                yC_f = np.reshape(yC, [-1, 1])
                x_f = np.reshape(x, -1, 1)
                x_flattened = np.concatenate((yC_f, x_f), axis=0)
        t = D
        y = net_Dmax(x_flattened)
        performance = perform(net_Dmax, t, y)

        # Save the trained model
        saver = tf.train.Saver()
        saver.save(sess, 'CBNND_max.ckpt')

    return WC_Dmax, net_Dmax, tr_Dmax


# Train the model
WC_Dmax, net_Dmax, tr_Dmax = model(X[:, :, 1:], D[0, 1:], hidden_layers, CC, epoch)

'''
# ### Вариант 2

# In[3]:

import scipy
import numpy as np
import tensorflow as tf
from keras import layers
#from tensorflow.keras import layers
from ReLU import ReLU
from Conv import Conv
from Pool import Pool
from trainConv import trainConv

# Load Data
Data = np.load('Data.npy')
Primary_data = np.load('Primary_data.npy')

D_Max = D_Max[D_Max != 'None']
X = Data[:, :, D_Max != 'None']

D = np.array(list(map(float, D_Max)))

# Initialize parameters
CC = [2, 2, 20]  # Cross correlation architecture 4*3*20
hidden_layers = [10, 10, 10]  # 3 layers
epoch = 100

def train_conv(X, D, hidden_layers, CC, epoch):
    model = tf.keras.Sequential([
        layers.Conv2D(CC[0], (3, 3), activation='relu'),
        layers.Conv2D(CC[1], (3, 3), activation='relu'),
        layers.Conv2D(CC[2], (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(hidden_layers[0], activation='relu'),
        layers.Dense(hidden_layers[1], activation='relu'),
        layers.Dense(hidden_layers[2], activation='relu'),
        layers.Dense(1)
    ])
    model.compile(optimizer='adam',
                  loss='mean_squared_error',
                  metrics=['mean_squared_error'])
    model.fit(X, D, epochs=epoch)
    return model

# Train Convolutional Neural Network
WC_Dmax = train_conv(X[:, :, 1:], D[1:], hidden_layers, CC, epoch)

x_flattened = []
for k in range(len(D)):
    # Data Correlating
    x = X[:, :, k]
    yC1 = WC_Dmax(x)  # assuming Conv is a function that applies the convolutional layers
    yC2 = tf.keras.activations.relu(yC1)
    yC = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(yC2)  # assuming Pool is a max pooling 2D layer
    # Data Flattening
    yC_f = tf.reshape(yC, [-1, 1])
    x_f = tf.reshape(x, [-1, 1])
    x_flattened.append(tf.concat([yC_f, x_f], axis=0))

x_flattened = tf.concat(x_flattened, axis=-1)

t = D
y = net_Dmax(x_flattened)


# Save model
WC_Dmax.save('CBNN_Dmax.h5')


# '''
