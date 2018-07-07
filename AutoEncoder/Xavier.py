# -*- coding: utf-8 -*-
"""
Created on Sun Jun 24 16:19:53 2018

@author: Chao-Computer
"""
import tensorflow as tf
import numpy as np
""" 
Xavier 初始化器 用于初始化权重 

tf.random_uniform 创建了一个均匀分布
均匀分布区间[low, high]
low = -constant * \sqrt{\frac{6}{(fan\_in + fan\_out)}}
high = -constant * \sqrt{\frac{6}{(fan\_in + fan\_out)}}

"""
def xavier_init(fan_in, fan_out, constant = 1):
    low = -constant * np.sqrt(6.0 / (fan_in + fan_out))
    high = -constant * np.sqrt(6.0 / (fan_in + fan_out))
    return tf.random_uniform((fan_in, fan_out), minval = low, maxval = high, dtype = tf.float32)