# FusionGAN TensorFlow 2 版本使用指南

本项目已从 TensorFlow 1.x 升级到 TensorFlow 2.x，以便在 macOS 系统上更好地运行，并进行了代码结构优化。

## 环境要求

- macOS 10.15 或更高版本
- Python 3.12 或更高版本
- TensorFlow 2.6.0 或更高版本

## 安装步骤

### 1. 安装依赖

项目使用pyproject.toml管理依赖，可以通过以下方式安装：

```bash
# 使用pip
pip install -e .

# 或使用uv（推荐）
uv pip install -e .
```

这将安装所有必要的依赖包，包括：
- tensorflow>=2.6.0
- numpy>=1.19.5
- scipy>=1.7.3
- h5py>=3.6.0
- matplotlib>=3.5.1
- Pillow>=9.0.1
- opencv-python>=4.5.5
- click>=8.0.0 (用于命令行接口)

### 2. 确保有合适的 Python 环境

建议使用虚拟环境来避免依赖冲突：

```bash
# 创建虚拟环境
python3 -m venv venv

# 激活虚拟环境
source venv/bin/activate

# 安装依赖
pip install -e .
```

## 项目结构

项目代码已整理到fusiongan目录中，结构如下：

```
FusionGAN/
├── fusiongan/
│   ├── model.py          # CGAN模型实现
│   ├── utils.py          # 工具函数集合
│   ├── train.py          # 模型训练脚本
│   └── test.py           # 图像融合测试脚本（支持命令行参数）
├── pyproject.toml        # 项目配置和依赖
├── README.md             # 项目说明文档
└── sample/               # 示例图像目录
```

## 使用方法

### 1. 训练模型

```bash
python fusiongan/train.py
```

训练配置可以在代码中直接修改。

### 2. 测试模型 - 使用命令行参数

`test.py` 已使用click库封装，可以通过命令行参数灵活配置：

```bash
python fusiongan/test.py --output-dir ./output --ir-dir ./data/ir --vis-dir ./data/vis
```

可用参数：
- `-o, --output-dir`: 输出文件夹路径（必需）
- `-i, --ir-dir`: 红外图像文件夹路径（必需）
- `-v, --vis-dir`: 可见光图像文件夹路径（必需）
- `-c, --checkpoint-dir`: 模型权重文件夹路径（默认：./checkpoint/CGAN_120）
- `-e, --epoch`: 要加载的模型训练轮数（默认：3）

示例：
```bash
# 基本用法
python fusiongan/test.py -o ./results -i ./dataset/infrared -v ./dataset/visible

# 指定检查点目录和训练轮数
python fusiongan/test.py -o ./results -i ./dataset/ir -v ./dataset/vis -c ./my_checkpoints -e 5
```

### 3. 高级用法

`test.py` 支持批量处理多对红外-可见光图像，程序会自动将同名称的图像进行配对融合。

## 在 macOS 上的特殊配置

### GPU 支持

TensorFlow 2 在 macOS 上支持 M1/M2 芯片的 GPU 加速。项目中的代码已包含自动启用 GPU 内存增长的配置：

```python
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("GPU 内存增长已启用")
    except RuntimeError as e:
        print(e)
```

这将确保 TensorFlow 不会立即占用所有 GPU 内存，而是根据需要动态增长。

### macOS 上的性能优化

1. 对于 M1/M2 芯片的 Mac，推荐使用 TensorFlow 官方提供的针对 Apple Silicon 优化的版本：

```bash
pip install tensorflow-macos tensorflow-metal
```

这将安装针对 macOS 优化的 TensorFlow 版本和 Metal 插件，以获得更好的 GPU 加速性能。

2. 可以适当调整 batch_size 参数以适应您的 Mac 硬件配置。

## 常见问题解决

### 1. 内存不足错误

如果遇到内存不足错误，可以尝试：
- 减小 batch_size 参数
- 使用更小的图像尺寸
- 关闭其他占用大量内存的应用程序

### 2. 模型权重加载失败

如果权重加载失败，请确保：
- 权重文件路径正确
- 使用了与训练时相同的模型架构
- TensorFlow 版本兼容

### 3. macOS 上的特定问题

对于 macOS 上的特定问题，可以参考 TensorFlow 官方文档：
https://www.tensorflow.org/install/macos

## 原始版本与 TensorFlow 2 版本的主要区别

1. **模型定义方式**：从 TensorFlow 1.x 的静态图定义改为 TensorFlow 2.x 的 Keras 函数式 API
2. **会话管理**：不再需要显式创建和管理 tf.Session
3. **优化器**：使用 tf.keras.optimizers 代替 tf.train.AdamOptimizer
4. **变量初始化**：不再需要显式调用 tf.initialize_all_variables()
5. **检查点管理**：使用 tf.train.Checkpoint 和 tf.train.CheckpointManager 代替 tf.train.Saver
6. **性能优化**：添加了 @tf.function 装饰器以提高性能

## 注意事项

1. 项目已完全迁移到TensorFlow 2.x版本，所有代码都使用Keras API实现
2. 代码结构已优化，所有功能文件都整理到fusiongan目录下
3. 权重文件路径可能需要根据您的具体环境进行调整
4. 首次运行时，程序会自动创建必要的目录结构

如有任何问题或建议，请随时提出反馈。