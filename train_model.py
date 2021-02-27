# -*- coding: UTF-8 -*-
# !/usr/bin/python
"""
@usage:
@author:qiaos
@file:train_model.py
@time:2021/02/26
@project:
"""
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os
import glob
from data_pretreat import read_images_png
from data_pretreat import read_masks_png
from data_pretreat import crop_img
from data_pretreat import load_train_image
from data_pretreat import load_test_image
#%%读取数据列表
#训练数据列表
#train数据集位置
images_train_filelist = glob.glob(r'CT_data\train\imgs\*.png')
masks_train_filelist = glob.glob(r'CT_data\train\masks\*.png')
train_count = len(images_train_filelist)
#重新排列数据的顺序
index = np.random.permutation(train_count)
images_train_filelist = np.array(images_train_filelist)[index]
masks_train_filelist = np.array(masks_train_filelist)[index]

#验证数据列表
#test数据集位置
images_test_filelist = glob.glob(r'CT_data\valid\imgs\*.png')
masks_test_filelist = glob.glob(r'CT_data\valid\masks\*.png')
test_count = len(images_test_filelist)

#%%

dataset_train = tf.data.Dataset.from_tensor_slices((images_train_filelist,masks_train_filelist))

#%%

dataset_test = tf.data.Dataset.from_tensor_slices((images_test_filelist,masks_test_filelist))

img_1 = read_images_png(images_train_filelist[0])

#%%

mask_1 = read_masks_png(masks_train_filelist[0])
#%%
if __name__ == '__main__':
    BATCH_SIZE = 5
    BUFFER_SIZE = 10
    step_per_epoch = train_count//BATCH_SIZE
    test_step = test_count//BATCH_SIZE
    auto = tf.data.experimental.AUTOTUNE

    dataset_train = dataset_train.map(load_train_image,num_parallel_calls = auto)
    dataset_test = dataset_test.map(load_test_image,num_parallel_calls = auto)

