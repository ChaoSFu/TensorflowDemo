#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul  7 14:47:12 2018

@author: fuchou
"""

from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
sess = tf.InteractiveSession()

# 定义网路结构
in_units = 784
h1_units = 300
out_units = 10

# 定义学习率
learning_rate = 0.3

# 定义参数结构
W1 = tf.Variable(tf.truncated_normal([in_units, h1_units], stddev=0.1))
b1 = tf.Variable(tf.zeros([h1_units]))

W2 = tf.Variable(tf.zeros([h1_units, out_units]))
b2 = tf.Variable(tf.zeros([out_units]))

# 定义变量结构
X = tf.placeholder(tf.float32, [None, in_units])
y_ = tf.placeholder(tf.float32, [None, 10])

# 定义dropout参数
keep_prob = tf.placeholder(tf.float32)

# 计算节点
hidden_out = tf.nn.relu(tf.matmul(X, W1) + b1)
hidden_drop = tf.nn.dropout(hidden_out, keep_prob)

# 定义输出
y = tf.nn.softmax(tf.matmul(hidden_drop, W2) + b2)

# 定义cost 和 优化方式
cost = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y),
                      reduction_indices=[1]))

train_step = tf.train.AdagradOptimizer(learning_rate).minimize(cost)

# 指定训练过程

## 初始化全局参数
tf.global_variables_initializer().run()

## 定义训练参数
total_train_size = 3000
sample_size_per_batch = 200
keep_prob_rate = 0.75

## 训练过程
for i in range(total_train_size):
    batch_xs, batch_ys = mnist.train.next_batch(sample_size_per_batch)
    train_step.run({X: batch_xs, y_:batch_ys, keep_prob: keep_prob_rate})
    
## 测试验证过程
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

print(accuracy.eval({X: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))