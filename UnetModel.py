# -*- coding: UTF-8 -*-
# !/usr/bin/python
"""
@usage:
@author:qiaos
@file:UnetModel.py
@time:2021/02/26
@project:
"""

import matplotlib.pyplot as plt
import numpy as np
import os
import glob
import tensorflow as tf


class DownsampleLayer(tf.keras.layers.Layer):
    def __int__(self, units):
        """
        units,cov2d的filters参数？ 这个参数的主要含义是什么？
        :param units:
        :return:
        """
        super(DownsampleLayer, self).__int__()
        self.conv1 = tf.keras.layers.Conv2D(units,
                                            kernel_size=3,
                                            padding='same')
        self.conv2 = tf.keras.layers.Conv2D(units,
                                            kernel_size=3,
                                            padding='same')
        self.pool = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))

    def call(self, inputs, is_pool=True):
        """
        layers类的call方法的重新定义？
        :param inputs:
        :param is_pool:
        :return:
        """
        if is_pool:
            inputs = self.pool(inputs)
        x = self.conv1(inputs)
        x = tf.nn.relu(x)
        x = self.conv2(x)
        outputs = tf.nn.relu(x)
        return outputs


class UpsampleLayer(tf.keras.layers.Layer):
    def __int__(self, units):
        super(UpsampleLayer, self).__int__()
        self.conv1 = tf.keras.layers.Conv2D(units, kernel_size=3,
                                            padding='same')
        self.conv2 = tf.keras.layers.Conv2D(units, kernel_size=3,
                                            padding='same')
        self.deconv = tf.keras.layers.Conv2DTranspose(units // 2,
                                                      kernel_size=2,
                                                      strides=2,
                                                      padding='same')

    def call(self, inputs):
        x = self.conv1(inputs)
        x = tf.nn.relu(x)
        x = self.conv2(x)
        x = tf.nn.relu(x)
        x = self.deconv(x)
        outputs = tf.nn.relu(x)
        return outputs


class UNetModel(tf.keras.Model):
    """
    定义模型，这个模型的父类是来自keras的Model类
    """

    def __int__(self):
        super(UNetModel, self).__init__()
        self.down_sample_I = DownsampleLayer(64)
        self.down_sample_II = DownsampleLayer(128)
        self.down_sample_III = DownsampleLayer(256)
        self.down_sample_IV = DownsampleLayer(512)
        self.down_sample_V = DownsampleLayer(1024)
        self.up_sample = tf.keras.layers.Conv2DTranspose(512,
                                                         kernel_size=2,
                                                         strides=2,
                                                         padding='same')
        self.up_sample_I = UpsampleLayer(512)
        self.up_sample_II = UpsampleLayer(256)
        self.up_sample_III = UpsampleLayer(128)
        self.conv_last = UpsampleLayer(64)
        self.last = tf.keras.layers.Conv2D(2,
                                           kernel_size=1,
                                           padding='same')



if __name__ == '__main__':
    pass

