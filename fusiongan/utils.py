# -*- coding: utf-8 -*-
import os
import glob
import h5py
import random
import matplotlib.pyplot as plt

from PIL import Image
import scipy.ndimage
import numpy as np
import tensorflow as tf
import cv2

# 在TensorFlow 2中，我们不再使用tf.app.flags
# 而是直接在主函数中定义配置参数

class Config:
    def __init__(self):
        self.epoch = 10
        self.batch_size = 32
        self.image_size = 132
        self.label_size = 120
        self.learning_rate = 1e-4
        self.c_dim = 1
        self.scale = 3
        self.stride = 14
        self.checkpoint_dir = "checkpoint"
        self.sample_dir = "sample"
        self.summary_dir = "log"
        self.is_train = True

# 全局配置对象，在需要时可以被覆盖
FLAGS = Config()

def read_data(path):
  """
  读取h5格式的数据文件
  """
  with h5py.File(path, 'r') as hf:
    data = np.array(hf.get('data'))
    label = np.array(hf.get('label'))
    return data, label

def preprocess(path, scale=3):
  """
  预处理单个图像文件
    (1) 以YCbCr格式读取原始图像（默认为灰度图）
    (2) 标准化
    (3) 对图像文件应用双三次插值
  """
  # 读取图片
  image = imread(path, is_grayscale=True)
  # 将图片label裁剪为scale的倍数
  label_ = modcrop(image, scale)

  # 必须标准化
  image = (image - 127.5) / 127.5 
  label_ = (image - 127.5) / 127.5 
  # 下采样之后再插值
  input_ = scipy.ndimage.interpolation.zoom(label_, (1./scale), prefilter=False)
  input_ = scipy.ndimage.interpolation.zoom(input_, (scale/1.), prefilter=False)

  return input_, label_

def prepare_data(dataset):
  """
  准备数据集
  """
  # 获取项目根目录
  root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
  
  if FLAGS.is_train:
    filenames = os.listdir(dataset)
    data_dir = os.path.join(root_dir, dataset)
    data = glob.glob(os.path.join(data_dir, "*.bmp"))
    data.extend(glob.glob(os.path.join(data_dir, "*.tif")))
    # 将图片按序号排序
    data.sort(key=lambda x: int(x[len(data_dir)+1:-4]))
  else:
    data_dir = os.path.join(root_dir, dataset)
    data = glob.glob(os.path.join(data_dir, "*.bmp"))
    data.extend(glob.glob(os.path.join(data_dir, "*.tif")))
    data.sort(key=lambda x: int(x[len(data_dir)+1:-4]))
  
  return data

def make_data(data, label, data_dir):
  """
  将输入数据制作为h5文件格式
  根据'is_train'（标志值），保存路径会更改
  """
  # 获取项目根目录
  root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
  
  if FLAGS.is_train:
    savepath = os.path.join(root_dir, os.path.join('checkpoint', data_dir, 'train.h5'))
    if not os.path.exists(os.path.join(root_dir, os.path.join('checkpoint', data_dir))):
        os.makedirs(os.path.join(root_dir, os.path.join('checkpoint', data_dir)))
  else:
    savepath = os.path.join(root_dir, os.path.join('checkpoint', data_dir, 'test.h5'))
    if not os.path.exists(os.path.join(root_dir, os.path.join('checkpoint', data_dir))):
        os.makedirs(os.path.join(root_dir, os.path.join('checkpoint', data_dir)))
  
  with h5py.File(savepath, 'w') as hf:
    hf.create_dataset('data', data=data)
    hf.create_dataset('label', data=label)

def imread(path, is_grayscale=True):
  """
  使用路径读取图像
  默认值为灰度图，图像按论文所述以YCbCr格式读取
  """
  if is_grayscale:
    # 以灰度图的形式读取
    img = Image.open(path).convert('L')
    return np.array(img).astype(np.float)
  else:
    img = Image.open(path).convert('YCbCr')
    return np.array(img).astype(np.float)

def modcrop(image, scale=3):
  """
  为了缩放原始图像，首先需要确保缩放操作没有余数
  我们需要找到高度（和宽度）与缩放因子的模
  然后，从原始图像大小的高度（和宽度）中减去模
  即使在缩放操作后也不会有余数
  """
  if len(image.shape) == 3:
    h, w, _ = image.shape
    h = h - np.mod(h, scale)
    w = w - np.mod(w, scale)
    image = image[0:h, 0:w, :]
  else:
    h, w = image.shape
    h = h - np.mod(h, scale)
    w = w - np.mod(w, scale)
    image = image[0:h, 0:w]
  return image

def input_setup(config, data_dir, index=0):
  """
  读取图像文件并制作其子图像，并将它们保存为h5文件格式
  """
  # 加载数据路径
  if config.is_train:
    # 取到所有的原始图片的地址
    data = prepare_data(dataset=data_dir)
  else:
    data = prepare_data(dataset=data_dir)

  sub_input_sequence = []
  sub_label_sequence = []
  padding = abs(config.image_size - config.label_size) // 2  # 6

  if config.is_train:
    for i in range(len(data)):
      input_ = (imread(data[i]) - 127.5) / 127.5
      label_ = input_

      if len(input_.shape) == 3:
        h, w, _ = input_.shape
      else:
        h, w = input_.shape
      # 按步长采样小patch
      for x in range(0, h - config.image_size + 1, config.stride):
        for y in range(0, w - config.image_size + 1, config.stride):
          sub_input = input_[x:x+config.image_size, y:y+config.image_size]  # [132 x 132]
          # 注意这里的padding，前向传播时由于卷积是没有padding的，所以实际上预测的是测试patch的中间部分
          sub_label = label_[x+padding:x+padding+config.label_size, y+padding:y+padding+config.label_size]  # [120 x 120]
          # 设置通道值
          sub_input = sub_input.reshape([config.image_size, config.image_size, 1])  
          sub_label = sub_label.reshape([config.label_size, config.label_size, 1])
          
          sub_input_sequence.append(sub_input)
          sub_label_sequence.append(sub_label)

  else:
    input_ = (imread(data[index]) - 127.5) / 127.5
    if len(input_.shape) == 3:
      h_real, w_real, _ = input_.shape
    else:
      h_real, w_real = input_.shape
    padding_h = config.image_size - ((h_real + padding) % config.label_size)
    padding_w = config.image_size - ((w_real + padding) % config.label_size)
    input_ = np.lib.pad(input_, ((padding, padding_h), (padding, padding_w)), 'edge')
    label_ = input_
    h, w = input_.shape
    # 计算合并操作所需的图像高度和宽度中的子图像数量
    nx = ny = 0 
    for x in range(0, h - config.image_size + 1, config.stride):
      nx += 1; ny = 0
      for y in range(0, w - config.image_size + 1, config.stride):
        ny += 1
        sub_input = input_[x:x+config.image_size, y:y+config.image_size]  # [132 x 132]
        sub_label = label_[x+padding:x+padding+config.label_size, y+padding:y+padding+config.label_size]  # [120 x 120]
        
        sub_input = sub_input.reshape([config.image_size, config.image_size, 1])  
        sub_label = sub_label.reshape([config.label_size, config.label_size, 1])

        sub_input_sequence.append(sub_input)
        sub_label_sequence.append(sub_label)

  # 将列表转换为numpy数组
  arrdata = np.asarray(sub_input_sequence)  # [?, 132, 132, 1]
  arrlabel = np.asarray(sub_label_sequence)  # [?, 120, 120, 1]
  
  make_data(arrdata, arrlabel, data_dir)

  if not config.is_train:
    return nx, ny, h_real, w_real
  else:
    return None

def imsave(image, path):
  """
  保存图像到指定路径
  """
  # 确保图像是在0-255范围内的uint8类型
  if image.dtype != np.uint8:
    if np.max(image) <= 1.0 and np.min(image) >= 0.0:
      image = (image * 255).astype(np.uint8)
    else:
      image = image.astype(np.uint8)
  
  # 处理灰度图和RGB图
  if len(image.shape) == 2 or (len(image.shape) == 3 and image.shape[2] == 1):
    # 灰度图
    if len(image.shape) == 3:
      image = image.squeeze()
    Image.fromarray(image, mode='L').save(path)
  else:
    # RGB图
    Image.fromarray(image, mode='RGB').save(path)

  return True

def merge(images, size):
  """
  将图像数组合并为一个大图像
  """
  h, w = images.shape[1], images.shape[2]
  img = np.zeros((h*size[0], w*size[1], 1))
  for idx, image in enumerate(images):
    i = idx % size[1]
    j = idx // size[1]
    img[j*h:j*h+h, i*w:i*w+w, :] = image

  return (img * 127.5 + 127.5)

def gradient(input_tensor):
  """
  计算输入张量的梯度
  在TensorFlow 2中使用tf.nn.conv2d
  """
  # 创建拉普拉斯滤波器
  filter = tf.constant([[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]], dtype=tf.float32)
  filter = tf.reshape(filter, [3, 3, 1, 1])
  
  # 确保输入有4个维度 (batch, height, width, channels)
  if len(input_tensor.shape) == 3:
    input_tensor = tf.expand_dims(input_tensor, axis=0)
  if input_tensor.shape[-1] != 1:
    input_tensor = tf.expand_dims(tf.reduce_mean(input_tensor, axis=-1), axis=-1)
  
  # 应用卷积计算梯度
  d = tf.nn.conv2d(input_tensor, filter, strides=[1, 1, 1, 1], padding='SAME')
  
  return d

def weights_spectral_norm(weights, u=None, iteration=1, update_collection=None, reuse=False, name='weights_SN'):
  """
  实现权重的谱归一化
  在TensorFlow 2中，我们使用tf.function和tf.Variable来实现
  """
  with tf.name_scope(name):
    w_shape = weights.shape.as_list()
    w_mat = tf.reshape(weights, [-1, w_shape[-1]])
    
    if u is None:
      u = tf.Variable(tf.random.truncated_normal([1, w_shape[-1]]), trainable=False)

    def power_iteration(u_var, ite):
      v_ = tf.matmul(u_var, tf.transpose(w_mat))
      v_hat = l2_norm(v_)
      u_ = tf.matmul(v_hat, w_mat)
      u_hat = l2_norm(u_)
      return u_hat, v_hat
    
    u_hat, v_hat = power_iteration(u, iteration)
    
    sigma = tf.matmul(tf.matmul(v_hat, w_mat), tf.transpose(u_hat))
    
    w_mat = w_mat / sigma
    
    if update_collection is None:
      u.assign(u_hat)
      w_norm = tf.reshape(w_mat, w_shape)
    else:
      if update_collection != 'NO_OPS':
        u.assign(u_hat)
      
      w_norm = tf.reshape(w_mat, w_shape)
    
    return w_norm

def lrelu(x, leak=0.2):
  """
  Leaky ReLU激活函数
  """
  return tf.maximum(x, leak * x)

def l2_norm(input_x, epsilon=1e-12):
  """
  L2归一化
  """
  input_x_norm = input_x / (tf.reduce_sum(input_x**2)**0.5 + epsilon)
  return input_x_norm