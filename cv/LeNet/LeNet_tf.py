#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：model_zoo 
@File ：LeNet_tf.py
@Author ：zhouhan
@Date ：2022/3/1 11:20 下午 
'''
from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model
import tensorflow as tf

# tensorflow 中的通道排序是 [B, H, W, C]
class Net(Model):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = Conv2D(filters=32, kernel_size=3, activation='relu') # tf的输入顺序和pytorch的不太一样
        self.flatten = Flatten() # 展平成一维
        self.d1 = Dense(128, activation='relu')
        self.d2 =Dense(10, activation='softmax')

    def call(self, x, **kwargs):
        x = self.conv1(x)
        print(x.shape)
        x = self.flatten(x)
        x = self.d1(x)
        x = self.d2(x)
        return x

if __name__ == '__main__':
    # [B, H, W, C]
    x = tf.random.uniform(shape=(1, 28, 28, 1))
    model = Net()
    out = model(x)
    print(out.shape)
