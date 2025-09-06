# -*- coding: utf-8 -*-
from .model import CGAN
from .utils import Config

import numpy as np
import tensorflow as tf
import pprint
import os
import sys

# 设置中文字体支持
plt = pprint.PrettyPrinter()

# 配置GPU内存增长，避免占用全部显存
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # 设置GPU内存增长
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("GPU 内存增长已启用")
    except RuntimeError as e:
        # 内存增长必须在程序开始时设置
        print(e)

class TrainConfig(Config):
    """
    训练配置类，继承自基本配置类
    可以在这里覆盖默认配置值
    """
    def __init__(self):
        super().__init__()
        # 覆盖默认值
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

# 命令行参数处理
if len(sys.argv) > 1:
    # 这里可以根据需要添加命令行参数解析
    pass

def main():
    # 创建配置对象
    config = TrainConfig()
    
    # 打印配置信息
    print("配置参数:")
    for attr in dir(config):
        if not attr.startswith('__') and not callable(getattr(config, attr)):
            print(f"  {attr}: {getattr(config, attr)}")

    # 创建必要的目录
    if not os.path.exists(config.checkpoint_dir):
        os.makedirs(config.checkpoint_dir)
    if not os.path.exists(config.sample_dir):
        os.makedirs(config.sample_dir)
    if not os.path.exists(config.summary_dir):
        os.makedirs(config.summary_dir)

    # 创建日志记录器
    train_summary_writer = tf.summary.create_file_writer(config.summary_dir + '/train')

    # 创建CGAN模型实例
    print("创建模型...")
    cgan = CGAN(
        image_size=config.image_size,
        label_size=config.label_size,
        batch_size=config.batch_size,
        c_dim=config.c_dim,
        checkpoint_dir=config.checkpoint_dir,
        sample_dir=config.sample_dir
    )

    print("开始训练...")
    # 开始训练
    cgan.train(config)
    
    print("训练完成!")

if __name__ == '__main__':
    main()