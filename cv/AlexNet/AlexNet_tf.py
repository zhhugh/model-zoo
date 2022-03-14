#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：model_zoo 
@File ：AlexNet_tf.py
@Author ：zhouhan
@Date ：2022/3/3 3:34 下午 
'''

import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model
import tensorflow.keras.layers as layers


class AlexNet(Model):
    def __init__(self, img_height, img_width):
        super(AlexNet, self).__init__()
        input_image = layers.Input(shape=(img_height, img_width, 3), dtype='float32') #[B, 224, 224, 3]
        x = layers.ZeroPadding2D(padding=((1, 2), (1, 2)))(input_image) # [B, 224, 224,



if __name__ == '__main__':
    x = tf.random.uniform(shape=(1, 32, 32, 1))
    x = layers.ZeroPadding2D((1, 2))(x)
    print(x)