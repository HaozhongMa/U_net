# -*- coding: UTF-8 -*-
# !/usr/bin/python
"""
@usage:
@author:qiaos
@file:data_pretreat.py
@time:2021/02/26
@project:
"""
import tensorflow as tf
import glob
import numpy as np
import matplotlib.pyplot as plt

#%%

#%%

def read_images_png(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_png(img,channels=3)
    return img

#%%
def read_masks_png(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_png(img,channels=1)
    return img

#%%



#%% md
#数据增强
def crop_img(img,mask):
    """
    叠放后裁剪、翻转
    :param img:
    :param mask:
    :return: (256,256,3),(256,256,1)
    """
    concat_img = tf.concat([img,mask],axis = -1)
    concat_img = tf.image.resize(concat_img,(280,280),
                                method = tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    crop_img = tf.image.random_crop(concat_img,[256,256,4])
    return crop_img[ :, :, :3], crop_img[ :, :, 3:]

#%%
def normal(img,mask):
    img = tf.cast(img, tf.float32)/127.5 - 1
    mask = tf.cast(mask/255, tf.int32)
    return img, mask

#%% md
#加载图片
def load_train_image(img_path,mask_path):
    img = read_images_png(img_path)
    mask = read_masks_png(mask_path)
    img,mask = crop_img(img,mask)
    if tf.random.uniform(()) > 0.5:
        img = tf.image.flip_left_right(img)
        mask = tf.image.flip_left_right(mask)
    img_f32,mask_int32 = normal(img,mask)
    return img_f32,mask_int32

def load_test_image(img_path,mask_path):
    """
    :param img_path:
    :param mask_path:
    :return:
    """
    img = read_images_png(img_path)
    mask = read_masks_png(mask_path)
    img = tf.image.resize(img,(256,256))
    mask = tf.image.resize(mask,(256,256))
    img_f32,mask_int32 = normal(img,mask)
    return img_f32, mask_int32


if __name__ == '__main__':
    img_1, mask_1 = crop_img(img_1, mask_1)

    # %%

    plt.subplot(1, 2, 1)
    plt.imshow(img_1.numpy())
    plt.subplot(1, 2, 2)
    plt.imshow(mask_1.numpy())
