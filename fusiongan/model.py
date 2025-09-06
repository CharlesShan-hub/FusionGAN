# -*- coding: utf-8 -*-
from .utils import (
  read_data, 
  input_setup, 
  imsave,
  merge,
  gradient,
  lrelu,
  weights_spectral_norm,
  l2_norm
)

import time
import os
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.utils import Progbar

class CGAN(object):

  def __init__(self, 
               image_size=132,
               label_size=120,
               batch_size=32,
               c_dim=1, 
               checkpoint_dir=None, 
               sample_dir=None):

    self.is_grayscale = (c_dim == 1)
    self.image_size = image_size
    self.label_size = label_size
    self.batch_size = batch_size
    self.c_dim = c_dim

    self.checkpoint_dir = checkpoint_dir
    self.sample_dir = sample_dir
    
    # 构建模型
    self.build_model()

  def build_model(self):
    # 创建输入层
    self.images_ir = layers.Input(shape=(self.image_size, self.image_size, self.c_dim), name='images_ir')
    self.labels_ir = layers.Input(shape=(self.label_size, self.label_size, self.c_dim), name='labels_ir')
    self.images_vi = layers.Input(shape=(self.image_size, self.image_size, self.c_dim), name='images_vi')
    self.labels_vi = layers.Input(shape=(self.label_size, self.label_size, self.c_dim), name='labels_vi')
    
    # 将红外和可见光图像在通道方向连起来
    self.input_image = layers.Concatenate(axis=-1)([self.images_ir, self.images_vi])
    
    # 创建融合模型
    self.fusion_model = self.create_fusion_model()
    self.fusion_image = self.fusion_model(self.input_image)
    
    # 创建判别器模型
    self.discriminator = self.create_discriminator()
    
    # 判别器对真实样本和生成样本的预测
    pos = self.discriminator(self.labels_vi)
    neg = self.discriminator(self.fusion_image)
    
    # 判别器损失
    pos_loss = tf.reduce_mean(tf.square(pos - tf.random.uniform(shape=[self.batch_size, 1], minval=0.7, maxval=1.2)))
    neg_loss = tf.reduce_mean(tf.square(neg - tf.random.uniform(shape=[self.batch_size, 1], minval=0, maxval=0.3)))
    self.d_loss = pos_loss + neg_loss
    
    # 生成器损失
    self.g_loss_1 = tf.reduce_mean(tf.square(neg - tf.random.uniform(shape=[self.batch_size, 1], minval=0.7, maxval=1.2)))
    self.g_loss_2 = tf.reduce_mean(tf.square(self.fusion_image - self.labels_ir)) + \
                    5 * tf.reduce_mean(tf.square(gradient(self.fusion_image) - gradient(self.labels_vi)))
    self.g_loss_total = self.g_loss_1 + 100 * self.g_loss_2
    
    # 为生成器和判别器分别创建优化器
    self.gen_optimizer = optimizers.Adam(learning_rate=1e-4)
    self.disc_optimizer = optimizers.Adam(learning_rate=1e-4)
    
    # 创建检查点管理器
    self.checkpoint = tf.train.Checkpoint(generator=self.fusion_model, 
                                         discriminator=self.discriminator,
                                         gen_optimizer=self.gen_optimizer,
                                         disc_optimizer=self.disc_optimizer)
    self.manager = tf.train.CheckpointManager(self.checkpoint, self.checkpoint_dir, max_to_keep=50)

  def create_fusion_model(self):
    # 使用函数式API创建融合模型
    inputs = layers.Input(shape=(self.image_size, self.image_size, 2))  # 2个通道：红外和可见光
    
    # 第一层
    x = layers.Conv2D(256, (5, 5), strides=(1, 1), padding='valid', use_bias=True, name='fusion_model/layer1/conv')(inputs)
    x = layers.BatchNormalization(epsilon=1e-5, momentum=0.9, name='fusion_model/layer1/bn')(x)
    x = layers.Lambda(lambda x: lrelu(x))(x)
    
    # 第二层
    x = layers.Conv2D(128, (5, 5), strides=(1, 1), padding='valid', use_bias=True, name='fusion_model/layer2/conv')(x)
    x = layers.BatchNormalization(epsilon=1e-5, momentum=0.9, name='fusion_model/layer2/bn')(x)
    x = layers.Lambda(lambda x: lrelu(x))(x)
    
    # 第三层
    x = layers.Conv2D(64, (3, 3), strides=(1, 1), padding='valid', use_bias=True, name='fusion_model/layer3/conv')(x)
    x = layers.BatchNormalization(epsilon=1e-5, momentum=0.9, name='fusion_model/layer3/bn')(x)
    x = layers.Lambda(lambda x: lrelu(x))(x)
    
    # 第四层
    x = layers.Conv2D(32, (3, 3), strides=(1, 1), padding='valid', use_bias=True, name='fusion_model/layer4/conv')(x)
    x = layers.BatchNormalization(epsilon=1e-5, momentum=0.9, name='fusion_model/layer4/bn')(x)
    x = layers.Lambda(lambda x: lrelu(x))(x)
    
    # 第五层
    outputs = layers.Conv2D(1, (1, 1), strides=(1, 1), padding='valid', use_bias=True, activation='tanh', name='fusion_model/layer5/conv')(x)
    
    model = models.Model(inputs=inputs, outputs=outputs, name='fusion_model')
    return model

  def create_discriminator(self):
    # 使用函数式API创建判别器模型
    inputs = layers.Input(shape=(self.label_size, self.label_size, self.c_dim))
    
    # 第一层
    x = layers.Conv2D(32, (3, 3), strides=(2, 2), padding='valid', use_bias=True, name='discriminator/layer_1/conv')(inputs)
    x = layers.Lambda(lambda x: lrelu(x))(x)
    
    # 第二层
    x = layers.Conv2D(64, (3, 3), strides=(2, 2), padding='valid', use_bias=True, name='discriminator/layer_2/conv')(x)
    x = layers.BatchNormalization(epsilon=1e-5, momentum=0.9, name='discriminator/layer_2/bn')(x)
    x = layers.Lambda(lambda x: lrelu(x))(x)
    
    # 第三层
    x = layers.Conv2D(128, (3, 3), strides=(2, 2), padding='valid', use_bias=True, name='discriminator/layer_3/conv')(x)
    x = layers.BatchNormalization(epsilon=1e-5, momentum=0.9, name='discriminator/layer_3/bn')(x)
    x = layers.Lambda(lambda x: lrelu(x))(x)
    
    # 第四层
    x = layers.Conv2D(256, (3, 3), strides=(2, 2), padding='valid', use_bias=True, name='discriminator/layer_4/conv')(x)
    x = layers.BatchNormalization(epsilon=1e-5, momentum=0.9, name='discriminator/layer_4/bn')(x)
    x = layers.Lambda(lambda x: lrelu(x))(x)
    
    # 展平并输出
    x = layers.Flatten()(x)
    outputs = layers.Dense(1, use_bias=True, name='discriminator/line_5/dense')(x)
    
    model = models.Model(inputs=inputs, outputs=outputs, name='discriminator')
    return model

  # 定义训练步骤
  @tf.function
  def train_step(self, batch_images_ir, batch_labels_ir, batch_images_vi, batch_labels_vi):
    # 训练判别器
    with tf.GradientTape() as disc_tape:
        # 计算判别器损失
        fusion_output = self.fusion_model(tf.concat([batch_images_ir, batch_images_vi], axis=-1))
        pos_output = self.discriminator(batch_labels_vi)
        neg_output = self.discriminator(fusion_output)
        
        pos_loss = tf.reduce_mean(tf.square(pos_output - tf.random.uniform(shape=[tf.shape(batch_images_ir)[0], 1], minval=0.7, maxval=1.2)))
        neg_loss = tf.reduce_mean(tf.square(neg_output - tf.random.uniform(shape=[tf.shape(batch_images_ir)[0], 1], minval=0, maxval=0.3)))
        d_loss = pos_loss + neg_loss
    
    # 计算判别器梯度并更新
    disc_gradients = disc_tape.gradient(d_loss, self.discriminator.trainable_variables)
    self.disc_optimizer.apply_gradients(zip(disc_gradients, self.discriminator.trainable_variables))
    
    # 训练生成器
    with tf.GradientTape() as gen_tape:
        # 计算生成器损失
        fusion_output = self.fusion_model(tf.concat([batch_images_ir, batch_images_vi], axis=-1))
        neg_output = self.discriminator(fusion_output)
        
        g_loss_1 = tf.reduce_mean(tf.square(neg_output - tf.random.uniform(shape=[tf.shape(batch_images_ir)[0], 1], minval=0.7, maxval=1.2)))
        g_loss_2 = tf.reduce_mean(tf.square(fusion_output - batch_labels_ir)) + \
                   5 * tf.reduce_mean(tf.square(gradient(fusion_output) - gradient(batch_labels_vi)))
        g_loss_total = g_loss_1 + 100 * g_loss_2
    
    # 计算生成器梯度并更新
    gen_gradients = gen_tape.gradient(g_loss_total, self.fusion_model.trainable_variables)
    self.gen_optimizer.apply_gradients(zip(gen_gradients, self.fusion_model.trainable_variables))
    
    return d_loss, g_loss_total

  def train(self, config):
    if config.is_train:
      input_setup(config, "Train_ir")
      input_setup(config, "Train_vi")
    else:
      nx_ir, ny_ir, h_real, w_real = input_setup(config, "Test_ir")
      nx_vi, ny_vi, _, _ = input_setup(config, "Test_vi")

    if config.is_train:
      data_dir_ir = os.path.join('./{}'.format(config.checkpoint_dir), "Train_ir", "train.h5")
      data_dir_vi = os.path.join('./{}'.format(config.checkpoint_dir), "Train_vi", "train.h5")
    else:
      data_dir_ir = os.path.join('./{}'.format(config.checkpoint_dir), "Test_ir", "test.h5")
      data_dir_vi = os.path.join('./{}'.format(config.checkpoint_dir), "Test_vi", "test.h5")

    train_data_ir, train_label_ir = read_data(data_dir_ir)
    train_data_vi, train_label_vi = read_data(data_dir_vi)
    
    # 尝试加载检查点
    if self.load():
      print(" [*] Load SUCCESS")
    else:
      print(" [!] Load failed...")

    if config.is_train:
      print("Training...")

      for ep in range(config.epoch):
        # 打乱数据
        indices = np.random.permutation(len(train_data_ir))
        train_data_ir = train_data_ir[indices]
        train_label_ir = train_label_ir[indices]
        train_data_vi = train_data_vi[indices]
        train_label_vi = train_label_vi[indices]
        
        # 创建进度条
        pb = Progbar(len(train_data_ir) // config.batch_size)
        
        # 按批次运行
        batch_idxs = len(train_data_ir) // config.batch_size
        for idx in range(0, batch_idxs):
          batch_images_ir = train_data_ir[idx*config.batch_size : (idx+1)*config.batch_size]
          batch_labels_ir = train_label_ir[idx*config.batch_size : (idx+1)*config.batch_size]
          batch_images_vi = train_data_vi[idx*config.batch_size : (idx+1)*config.batch_size]
          batch_labels_vi = train_label_vi[idx*config.batch_size : (idx+1)*config.batch_size]

          # 训练两步判别器，一步生成器
          for i in range(2):
            d_loss, _ = self.train_step(batch_images_ir, batch_labels_ir, batch_images_vi, batch_labels_vi)
          _, g_loss = self.train_step(batch_images_ir, batch_labels_ir, batch_images_vi, batch_labels_vi)

          pb.update(idx + 1, [('loss_d', d_loss), ('loss_g', g_loss)])

        # 保存模型
        self.save(ep)

    else:
      print("Testing...")

      # 进行预测
      result = self.fusion_model.predict(tf.concat([train_data_ir, train_data_vi], axis=-1))
      result = result * 127.5 + 127.5
      result = merge(result, [nx_ir, ny_ir])
      result = result.squeeze()
      image_path = os.path.join(os.getcwd(), config.sample_dir)
      image_path = os.path.join(image_path, "test_image.png")
      imsave(result, image_path)

  def save(self, step):
    model_dir = "%s_%s" % ("CGAN", self.label_size)
    checkpoint_dir = os.path.join(self.checkpoint_dir, model_dir)

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    self.manager.save(checkpoint_number=step)

  def load(self):
    print(" [*] Reading checkpoints...")
    
    if self.manager.latest_checkpoint:
        self.checkpoint.restore(self.manager.latest_checkpoint)
        print(f" [*] Restored from {self.manager.latest_checkpoint}")
        return True
    else:
        print(" [*] No checkpoint found")
        return False