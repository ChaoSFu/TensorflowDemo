# -*- coding: utf-8 -*-
"""
Created on Sat Jun 23 18:25:40 2018

@author: Chao-Computer
"""
import tensorflow as tf
import numpy as np
from Xavier import xavier_init

class AdditiveGaussianNoiseAutoencoder(object):
    """
    n_input : 输入节点数
    n_hidden : 隐藏层节点数
    transfer_function : 隐含层激活函数，默认是softplus
    optimizer : 优化器，默认是Adam
    scale : 高斯噪音系数，默认是0.1
    当前使用一个隐含层
    """

    def __init__(self, n_input, n_hidden, transfer_function=tf.nn.softplus, optimizer = tf.train.AdamOptimizer(), scale = 0.1):
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.transfer = transfer_function
        self.scale = tf.placeholder(tf.float32)
        self.training_scale = scale
        
        """ 参数初始化 """
        network_weights = self._initialize_weights()
        self.weights = network_weights
        
        """ 定义输入变量X结构 """
        self.x = tf.placeholder(tf.float32, [None, self.n_input])
        
        """ 定义隐藏层结构 hidden = transfer(w1 * (x + scale * random\_normal((input,))) + b1) """
        self.hidden = self.transfer(tf.add(tf.matmul(self.x + scale * tf.random_normal((n_input,)), self.weights['w1']), self.weights['b1']))
        
        """ 定义输出层结构 reconstruction = w2 * hidden + b2 """
        self.reconstruction = tf.add(tf.matmul(self.hidden, self.weights['w2']), self.weights['b2'])
        
        """ 定义损失函数结构 cost = (reconstruction - x)^{2.0} """
        self.cost = 0.5 * tf.reduce_sum(tf.pow(tf.subtract(self.reconstruction, self.x), 2.0))
        
        """ 定义优化方法 optimizer = optimizer.minimize(cost) """
        self.optimizer = optimizer.minimize(self.cost)
        
        """ 初始化输入、目标变量 """
        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)
        
        
    """
    初始化参数
    w1 [n_input, n_hidden]
    b1 [n_hidden]
    
    w2 [n_hidden, n_output]
    b2 [n_output]
    
    n_input == out_put
    
    """
    def _initialize_weights(self):
        all_weights = dict()
        all_weights['w1'] = tf.Variable(xavier_init(self.n_input, self.n_hidden))
        all_weights['b1'] = tf.Variable(tf.zeros([self.n_hidden], dtype = tf.float32))
        all_weights['w2'] = tf.Variable(tf.zeros([self.n_hidden,self.n_input], dtype = tf.float32))
        all_weights['b2'] = tf.Variable(tf.zeros([self.n_input], dtype = tf.float32))
        
        return all_weights
        
    """ 
    分别计算两个节点
    cost 定义的损失函数
    optimizer 优化过程
    
    输入了两个placeholder
    X
    training_scale
    """
    def partial_fit(self, X):
        cost, opt = self.sess.run((self.cost, self.optimizer), feed_dict = {self.x: X, self.scale: self.training_scale})
        return cost
    
    """
    单独计算
    cost 定义的损失函数
    """
    def calc_total_cost(self, X):
        return self.sess.run(self.cost, feed_dict = {self.x: X, self.scale: self.training_scale})
    
    """
    单独计算
    hidden 隐藏层输出结果
    """
    def transform(self, X):
        return self.sess.run(self.hidden, feed_dict = {self.x: X, self.scale: self.training_scale})
    
    """
    利用输入的hidden，如果hidden输入为空，np.random.normal(size = self.weights['b1'])
    计算节点
    reconstruction = w2 * hidden + b2
    """
    def generate(self, hidden = None):
        if hidden is None:
            hidden = np.random.normal(size = self.weights['b1'])
        return self.sess.run(self.reconstruction, feed_dict = {self.hidden: hidden})
    
    """
    利用输入的X和training_scale计算
    计算节点
    reconstruction = w2 * hidden + b2
    """
    def reconstruct(self, X):
        return self.sess.run(self.reconstruction, feed_dict = {self.x: X, self.scale: self.training_scale})
    
    """
    计算w1权值
    """
    def getWeights(self):
        return self.sess.run(self.weights['w1'])
    
    """
    计算b1权值
    """
    def getBiases(self):
        return self.sess.run(self.weights['b1'])