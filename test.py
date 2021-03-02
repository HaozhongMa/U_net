# -*- coding: UTF-8 -*-
# !/usr/bin/python
"""
@usage:
@author:qiaos
@file:test.py
@time:2021/03/02
@project:
"""
# %% import

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os
import glob

IMAGE_DIR=r'c:\work\unetct\ct_data'
__debug__ == True
print("debug=%s" % __debug__)


# %% 定义UnetModel类

class DownsampleLayer(tf.keras.layers.Layer):
    conv1 = tf.keras.layers.Conv2D(64, kernel_size=3, padding='same')
    conv2 = tf.keras.layers.Conv2D(64, kernel_size=3, padding='same')
    pool = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))

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


class UnetModel(tf.keras.Model):
    """
    定义模型，这个模型的父类是来自keras的Model类
    """

    def __int__(self):
        super(UnetModel, self).__init__()
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

    def call(self, inputs):
        # 下采样5层
        x1 = self.down_sample_I(inputs, is_pool=False)
        x2 = self.down_sample_II(x1)
        x3 = self.down_sample_III(x2)
        x4 = self.down_sample_IV(x3)
        x5 = self.down_sample_V(x4)
        # 上采样5层
        x5 = self.up_sample(x5)
        x5 = tf.concat([x4, x5])
        x5 = self.up_sample_I(x5)
        x5 = tf.concat([x3, x5])
        x5 = self.up_sample_II(x5)
        x5 = tf.concat([x2, x5])
        x5 = self.up_sample_III(x5)
        x5 = tf.concat([x1, x5])
        x5 = self.conv_last(x5, is_pool=False)
        x5 = self.last(x5)
        return x5


# %% 定义数据预处理函数
def read_images_png(image_filename):
    """
    读取数据文件
    :param path:
    :return:
    """
    img = tf.io.read_file(image_filename)
    img = tf.image.decode_png(img, channels=3)
    return img


def read_masks_png(path):
    """
    读取mask文件
    :param path:
    :return: (rows,cols,1)
    """
    img = tf.io.read_file(path)
    img = tf.image.decode_png(img, channels=1)
    return img


# 数据增强
def crop_img(img, mask):
    """
    叠放后裁剪、翻转
    :param img:
    :param mask:
    :return: (256,256,3),(256,256,1)
    """
    concat_img = tf.concat([img, mask], axis=-1)
    concat_img = tf.image.resize(concat_img, (280, 280),
                                 method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    cropped_img = tf.image.random_crop(concat_img, [256, 256, 4])
    print("shape of cropped_img is ", cropped_img.shape)
    return cropped_img[:, :, :3], cropped_img[:, :, 3:]


def normal(img, mask):
    """
    对数据进行正态化处理，让值成为[-127.5,0,127.5]的分布状态
    :param img:
    :param mask:
    :return: img(rows,cols,layers=3)(float32),mask(rows,cols,layers=1)(int32)
    """
    img = tf.cast(img, tf.float32) / 127.5 - 1
    mask = tf.cast(mask / 255, tf.int32)
    return img, mask


def load_train_image(img_path, mask_path):
    """
    读取训练数据集，将图像和mask 合并以后，进行处理
    :param img_path:
    :param mask_path:
    :return:
    """
    # 读取数据
    img = read_images_png(img_path)
    mask = read_masks_png(mask_path)
    # 裁剪
    img, mask = crop_img(img, mask)
    # 翻转
    if tf.random.uniform(()) > 0.5:
        img = tf.image.flip_left_right(img)
        mask = tf.image.flip_left_right(mask)
    img_f32, mask_int32 = normal(img, mask)
    return img_f32, mask_int32


def load_test_image(img_path, mask_path):
    """
    读取测试数据集，这个函数
    todo:修改错误的部分，没有经过280,280的压缩，恐怕会有问题
    :param img_path:
    :param mask_path:
    :return:
    """
    # 读取数据
    img = read_images_png(img_path)
    mask = read_masks_png(mask_path)
    # 直接压缩数据到（256,256,3），（256,256,1）
    img = tf.image.resize(img, (256, 256))
    mask = tf.image.resize(mask, (256, 256))
    img_f32, mask_int32 = normal(img, mask)
    return img_f32, mask_int32


if __debug__ == True:
    img_path = r"c:\work\unetct\ct_data\train\imgs\0.png"
    mask_path = r"c:\work\unetct\ct_data\train\imgs\0.png"
    img, mask = load_test_image(img_path, mask_path)
    plt.imshow(img)
    plt.show()



# %%训练
# 训练数据列表
# train数据集位置,当前逻辑上存在风险。如果文件名称排序方式有差异的话，会错误。
# 增加了文件数量验证，不完备
images_train_filelist = glob.glob(IMAGE_DIR+r'\train\imgs\*.png')
masks_train_filelist = glob.glob(IMAGE_DIR+r'\train\masks\*.png')
if len(images_train_filelist)==len(masks_train_filelist):
    train_count = len(images_train_filelist)
    print("Train set has %d samples"%train_count)
else:
    print("Error source data for validation! images=%d,masks=%d"%(len(images_train_filelist),len(masks_train_filelist)))


# 重新排列数据的顺序
index = np.random.permutation(train_count)
images_train_filelist = np.array(images_train_filelist)[index]
masks_train_filelist = np.array(masks_train_filelist)[index]

# 验证数据列表
# test数据集位置,当前逻辑上存在风险。如果文件名称排序方式有差异的话，会错误。
# 增加了文件数量验证，不完备
images_test_filelist = glob.glob(IMAGE_DIR+r'\valid\imgs\*.png')
masks_test_filelist = glob.glob(IMAGE_DIR+r'\valid\masks\*.png')
if len(images_test_filelist)==len(masks_test_filelist):
    test_count = len(images_test_filelist)
    print("Validation set has %d samples"%test_count)
else:
    print("Error source data for validation! images=%d,masks=%d"%(len(images_test_filelist),len(masks_test_filelist)))


#%%定义数据集
# 定义训练参数
EPOCHS = 5
BATCH_SIZE = 5
BUFFER_SIZE = 10
step_per_epoch = train_count // BATCH_SIZE
test_step = test_count // BATCH_SIZE
auto = tf.data.experimental.AUTOTUNE

"""思路
Step0: 准备要加载的数据
Step1: 使用 tf.data.Dataset.from_tensor_slices() 函数进行加载
Step2: 使用 map() 函数进行预处理
Step3: 使用 shuffle() 打乱数据
Step4: 使用 batch() 函数设置 batch size 值
Step5: 根据需要 使用 repeat() 设置是否循环迭代数据集
"""
# Step1: 使用 tf.data.Dataset.from_tensor_slices() 函数进行加载dataset
dataset_train = tf.data.Dataset.from_tensor_slices((images_train_filelist, masks_train_filelist))  # step1,4,5
dataset_test = tf.data.Dataset.from_tensor_slices((images_test_filelist, masks_test_filelist))
# Step2: 使用 map() 函数读入数据
dataset_train = dataset_train.map(load_train_image, num_parallel_calls=auto)
dataset_test = dataset_test.map(load_test_image, num_parallel_calls=auto)
if __debug__ == True:
    for i, m in dataset_train.take(1):
        print(i.shape, m.shape)
        plt.subplot(1, 2, 1)
        plt.imshow((i.numpy() + 1) / 2)
        plt.subplot(1, 2, 2)
        plt.imshow(np.squeeze(m.numpy()))
        plt.show()

# Step3: 使用 shuffle() 打乱数据
dataset_train = dataset_train.cache().repeat().shuffle(BUFFER_SIZE)
# Step4: 使用 batch() 函数设置 batch size 值
dataset_train=dataset_train.batch(BATCH_SIZE).prefetch(auto)
dataset_test = dataset_test.cache().batch(BATCH_SIZE)
# Step5: 根据需要 使用 repeat() 设置是否循环迭代数据集

# %%训练步骤

model = UnetModel()

# %%定义训练过程中使用的函数

# 定义损失函数
class MeanIoU(tf.keras.metrics.MeanIoU):
    def __call__(self, y_ture, y_pred, sample_weight=None):
        y_pred = tf.argmax(y_pred, axis=-1)
        return super().__call__(y_ture, y_pred, sample_weight=sample_weight)


# 定义训练步骤
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_acc')
train_iou = MeanIoU(2, name='train_iou')

test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_acc')
test_iou = MeanIoU(2, name='test_iou')


@tf.function
def train_step(images, labels):
    with tf.GradientTape() as tape:
        predictions = model(images)
        # 需要看看这步有没有问题，这里变量名称定义的有点混乱
        loss = loss(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss(loss)
    train_accuracy(labels, predictions)
    train_iou(labels, predictions)


@tf.function
def test_step(images, labels):
    predictions = model(images)
    t_loss = loss_object(labels, predictions)
    test_loss(t_loss)

    test_accuracy(labels, predictions)
    test_iou(labels, predictions)


for epoch in range(EPOCHS):
    # 训练一个epoch的数据
    train_loss.reset_states()
    train_accuracy.reset_states()
    train_iou.reset_states()
    test_loss.reset_states()
    test_accuracy.reset_states()
    test_iou.reset_states()
    # 训练
    for images, labels in dataset_train:  # 这个应该是一张图片放进去训练吧
        train_step(images, labels)
    # 验证
    for test_images, test_labels in dataset_test:
        train_step(test_images, test_labels)
    # 输出
    template = 'EPOCH {:.3f}, Loss:{:.3f}, Accuracy: {:.3f},\
               IOU:{:.3f},Test Loss:{:.3f},\
               Test Acurracy:{:.3f},Test IOU:{:.3f}'

    print(template.format(epoch + 1,
                          train_loss.result(),
                          train_accuracy.result() * 100,
                          train_iou.result(),
                          test_loss.result(),
                          test_accuracy.result() * 100,
                          test_iou.result()
                          ))



