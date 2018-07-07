#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 23 16:18:45 2018

@author: fuchou
"""
from AdditiveGaussianNoiseAutoencoder import AdditiveGaussianNoiseAutoencoder
from tensorflow.examples.tutorials.mnist import input_data
import sklearn.preprocessing as prep
import numpy as np
import tensorflow as tf
"""
对训练数据和测试数据进行标准化处理
----->均值为0，标准差为1的分布
(处理方法，减去均值，再除以标准差)
训练数据与测试数据使用相同的scale
"""
def standard_scale(X_train, X_test):
    preprocessor = prep.StandardScaler().fit(X_train)
    X_train = preprocessor.transform(X_train)
    X_test = preprocessor.transform(X_test)
    
    return X_train, X_test
    
"""
随机获取数据
"""
def get_random_block_from_data(data, batch_size):
    start_index = np.random.randint(0, len(data) - batch_size)
    return data[start_index:(start_index + batch_size)]

mnist = input_data.read_data_sets("MNIST_DATA", one_hot = True)
X_train, X_test = standard_scale(mnist.train.images, mnist.test.images)

n_samples = int(mnist.train.num_examples)
training_epochs = 20
batch_size = 128
display_step = 1

autoencoder = AdditiveGaussianNoiseAutoencoder(n_input = 784, 
                                               n_hidden = 200,
                                               transfer_function = tf.nn.softplus,
                                               optimizer = tf.train.AdamOptimizer(learning_rate = 0.001),
                                               scale = 0.01)

for epoch in range(training_epochs):
    avg_cost = 0.
    total_batch = int(n_samples / batch_size)
    for i in range(total_batch):
        batch_xs = get_random_block_from_data(X_train, batch_size)
        
        cost = autoencoder.calc_total_cost(batch_xs)
        avg_cost += cost / n_samples * batch_size
        
    if epoch % display_step == 0 :
        print("Epoch:", '%04d' % (epoch + 1), "cost=", "{.9f}".format(avg_cost))
        

print("Total cost: " + str(autoencoder.calc_total_cost(X_test)))
    