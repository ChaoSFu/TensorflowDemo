#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 23 14:20:16 2018

@author: fuchou
"""

""" 获取数据集 """
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot = True)

print(mnist.train.images.shape, mnist.train.labels.shape)
print(mnist.test.images.shape, mnist.test.labels.shape)
print(mnist.validation.images.shape, mnist.validation.labels.shape)


""" 初始化参数 """
import tensorflow as tf
sess = tf.InteractiveSession()
x = tf.placeholder(tf.float32, [None, 784])

W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))


""" 构建目标函数 """
y = tf.nn.softmax(tf.matmul(x, W) + b)


""" 构建损失函数 """
y_ = tf.placeholder(tf.float32, [None, 10])
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))


""" 定义优化方法，梯度下降，步长0.5，最小化损失函数 """
train_step = tf.train.GradientDescentOptimizer(0.78).minimize(cross_entropy)


""" 初始化全局参数 """
tf.global_variables_initializer().run()


""" 训练过程，1000 次训练，每次 100 个样本 """
for i in range(500):
    batch_xs, batch_ys = mnist.train.next_batch(200)
    train_step.run({x: batch_xs, y_: batch_ys})
    
    
""" 构建准确率计算方法 """
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))  # tf.cast 参数类型转化  


""" 计算并输出准确率 """
print(accuracy.eval({x: mnist.test.images, y_: mnist.test.labels}))