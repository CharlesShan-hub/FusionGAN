# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import time
import os
import glob
import re
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import click
from tqdm import tqdm

# 设置中文字体支持
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]

# 定义Leaky ReLU激活函数
def lrelu(x, leak=0.2):
    return tf.maximum(x, leak * x)

# 图像读取函数
def imread(path, is_grayscale=False):
    """
    读取图像文件
    """
    if is_grayscale:
        img = Image.open(path).convert('L')
        return np.array(img).astype(float)
    else:
        img = Image.open(path).convert('YCbCr')
        return np.array(img).astype(float)

# 图像保存函数
def imsave(image, path):
    """
    保存图像文件
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
        img = Image.fromarray(image)
        img = img.convert('L')
        img.save(path)
    else:
        # RGB图
        img = Image.fromarray(image)
        img = img.convert('RGB')
        img.save(path)
    
    return True

# 数据准备函数
def prepare_data(dataset_path):
    """
    准备数据集，支持非纯数字文件名
    """
    data = glob.glob(os.path.join(dataset_path, "*.png"))
    
    # 自定义排序函数，处理非纯数字文件名
    def sort_key(file_path):
        base_name = os.path.basename(file_path).split('.')[0]
        # 尝试提取文件名中的数字部分
        try:
            # 检查是否全部是数字
            return (0, int(base_name))
        except ValueError:
            # 如果不是全部数字，尝试提取开头的数字
            match = re.match(r'(\d+)(.*)', base_name)
            if match:
                num_part = int(match.group(1))
                rest_part = match.group(2)
                return (1, num_part, rest_part)
            else:
                # 如果没有数字部分，直接按字符串排序
                return (2, base_name)
    
    # 使用自定义排序函数
    data.sort(key=sort_key)
    return data

# 创建融合模型
def create_fusion_model():
    """
    创建融合模型结构，与原始模型保持一致
    """
    # 定义输入层
    inputs = tf.keras.Input(shape=(None, None, 2))  # 2个通道：红外和可见光
    
    # 第一层
    x = tf.keras.layers.Conv2D(256, (5, 5), strides=(1, 1), padding='valid', use_bias=True, name='layer1_conv')(inputs)
    x = tf.keras.layers.BatchNormalization(epsilon=1e-5, momentum=0.9, name='layer1_bn')(x)
    x = tf.keras.layers.Lambda(lambda x: lrelu(x))(x)
    
    # 第二层
    x = tf.keras.layers.Conv2D(128, (5, 5), strides=(1, 1), padding='valid', use_bias=True, name='layer2_conv')(x)
    x = tf.keras.layers.BatchNormalization(epsilon=1e-5, momentum=0.9, name='layer2_bn')(x)
    x = tf.keras.layers.Lambda(lambda x: lrelu(x))(x)
    
    # 第三层
    x = tf.keras.layers.Conv2D(64, (3, 3), strides=(1, 1), padding='valid', use_bias=True, name='layer3_conv')(x)
    x = tf.keras.layers.BatchNormalization(epsilon=1e-5, momentum=0.9, name='layer3_bn')(x)
    x = tf.keras.layers.Lambda(lambda x: lrelu(x))(x)
    
    # 第四层
    x = tf.keras.layers.Conv2D(32, (3, 3), strides=(1, 1), padding='valid', use_bias=True, name='layer4_conv')(x)
    x = tf.keras.layers.BatchNormalization(epsilon=1e-5, momentum=0.9, name='layer4_bn')(x)
    x = tf.keras.layers.Lambda(lambda x: lrelu(x))(x)
    
    # 第五层
    outputs = tf.keras.layers.Conv2D(1, (1, 1), strides=(1, 1), padding='valid', use_bias=True, activation='tanh', name='layer5_conv')(x)
    
    # 创建模型
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name='fusion_model')
    
    return model

# 输入预处理函数
def input_setup(index, data_ir, data_vi, is_grayscale=False):
    """
    预处理输入图像
    """
    padding = 6
    
    # 处理红外图像
    input_ir = (imread(data_ir[index], is_grayscale=True) - 127.5) / 127.5
    input_ir = np.pad(input_ir, ((padding, padding), (padding, padding)), 'edge')
    h, w = input_ir.shape
    input_ir = input_ir.reshape([1, h, w, 1])  # 添加批次维度和通道维度
    
    # 处理可见光图像
    input_vi = (imread(data_vi[index], is_grayscale=is_grayscale) - 127.5) / 127.5
    if is_grayscale:
        input_vi = np.pad(input_vi, ((padding, padding), (padding, padding)), 'edge')
        h, w = input_vi.shape
        input_vi = input_vi.reshape([1, h, w, 1])  # 添加批次维度和通道维度
    else:
        input_vi = np.pad(input_vi, ((padding, padding), (padding, padding), (0, 0)), 'edge')
        h, w, _ = input_vi.shape
        input_vi = input_vi.reshape([1, h, w, 3])  # 添加批次维度
    
    return input_ir, input_vi

# 主函数
@click.command()
@click.option('--output-dir', '-o', required=True, type=click.Path(), 
              help='输出文件夹路径')
@click.option('--ir-dir', '-i', required=True, type=click.Path(exists=True), 
              help='红外图像文件夹路径')
@click.option('--vis-dir', '-v', required=True, type=click.Path(exists=True), 
              help='可见光图像文件夹路径')
@click.option('--checkpoint-dir', '-c', default='./checkpoint/CGAN_120', type=click.Path(), 
              help='模型权重文件夹路径')
@click.option('--epoch', '-e', default=3, type=int, 
              help='要加载的模型训练轮数')
def main(output_dir, ir_dir, vis_dir, checkpoint_dir, epoch):
    # 设置GPU内存增长，避免占用全部显存
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # 设置GPU内存增长
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"GPU 可用，使用GPU模式: {len(gpus)}个GPU设备")
        except RuntimeError as e:
            # 内存增长必须在程序开始时设置
            print(e)
    else:
        print("没有检测到GPU设备，自动切换到CPU模式")
        # 强制使用CPU
        tf.config.set_visible_devices([], 'GPU')
    
    # 创建结果目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 准备测试数据
    print("准备测试数据...")
    data_ir = prepare_data(ir_dir)
    data_vi = prepare_data(vis_dir)
    
    print(f"找到 {len(data_ir)} 对红外-可见光图像")
    
    # 创建模型
    print("创建融合模型...")
    fusion_model = create_fusion_model()
    
    # 打印模型结构
    fusion_model.summary()
    
    # 创建检查点
    checkpoint_path = f'{checkpoint_dir}/CGAN.model-{epoch}'
    print(f"加载权重文件: {checkpoint_path}")
    
    # 创建检查点对象
    checkpoint = tf.train.Checkpoint(model=fusion_model)
    
    # 尝试加载权重
    try:
        # 首先尝试使用CheckpointManager查找最新的检查点
        manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=5)
        
        if manager.latest_checkpoint:
            checkpoint.restore(manager.latest_checkpoint)
            print(f"成功加载权重: {manager.latest_checkpoint}")
        else:
            # 尝试直接从指定路径加载
            checkpoint.restore(checkpoint_path)
            print(f"成功加载权重: {checkpoint_path}")
    except Exception as e:
        print(f"无法加载权重: {e}")
        print("请检查权重文件路径是否正确")
        return
    
    # 处理图像
    print("开始处理图像...")
    total_time = 0
    is_grayscale = False
    
    # 使用tqdm创建进度条
    for i in tqdm(range(len(data_ir)), desc="处理进度", unit="张", ascii=True):
        start = time.time()
        
        # 预处理输入图像
        train_data_ir, train_data_vi = input_setup(i, data_ir, data_vi, is_grayscale=is_grayscale)
        
        # 进行融合
        if is_grayscale == False:
            # 使用可见光图像的Y通道
            # 创建连接的输入
            input_image = np.concatenate([train_data_ir, train_data_vi[:, :, :, 0:1]], axis=-1)
            result = fusion_model.predict(input_image)
        else:
            # 创建连接的输入
            input_image = np.concatenate([train_data_ir, train_data_vi], axis=-1)
            result = fusion_model.predict(input_image)
        
        # 后处理结果
        result = result * 127.5 + 127.5
        result = result.squeeze()
        result = (result - result.min()) / (result.max() - result.min()) * 255.0
        
        # 如果不是灰度图，替换YCbCr的Y通道
        if is_grayscale == False:
            temp = train_data_vi * 127.5 + 127.5
            temp[0, 6:-6, 6:-6, 0] = result  # 替换 Y 通道
            temp = temp[0, 6:-6, 6:-6, :]
            # 转换为RGB
            # Pillow 13弃用了mode参数，使用先创建Image再转换的方式
            img_ycbcr = Image.fromarray(temp.astype(np.uint8))
            img_rgb = img_ycbcr.convert('RGB')
            temp2 = np.asarray(img_rgb)
        
        # 保存结果
        image_path = Path(output_dir) / Path(data_ir[i]).name
        imsave(result, image_path)
        
        end = time.time()
        elapsed_time = end - start
        total_time += elapsed_time
    
    print(f"所有图像处理完成!")
    print(f"总耗时: {total_time:.4f}秒")
    print(f"平均每张耗时: {total_time/len(data_ir):.4f}秒")

if __name__ == '__main__':
    main()