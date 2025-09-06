#!/bin/bash

# 项目根目录
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(dirname "$script_dir")"

# 设置路径参数
output_dir="/Volumes/Charles/data/vision/torchvision/msrs/test/fused/fusiongan"
ir_dir="/Volumes/Charles/data/vision/torchvision/msrs/test/ir"
vis_dir="/Volumes/Charles/data/vision/torchvision/msrs/test/vi"
checkpoint_dir="$project_root/checkpoint/CGAN_120"
epoch=3

# 检查输入目录是否存在
for dir in "$ir_dir" "$vis_dir"; do
    if [ ! -d "$dir" ]; then
        echo "错误: 目录 '$dir' 不存在!"
        exit 1
    fi
done

# 创建输出目录
mkdir -p "$output_dir"

# 显示运行信息
echo "开始运行图像融合..."
echo "- 输出目录: $output_dir"
echo "- 红外图像目录: $ir_dir"
echo "- 可见光图像目录: $vis_dir"
echo ""

# 检查Python脚本是否存在
test_script="$project_root/fusiongan/test.py"
if [ ! -f "$test_script" ]; then
    echo "错误: 测试脚本 '$test_script' 不存在!"
    exit 1
fi

# 运行测试脚本
python "$test_script" \
    --output-dir "$output_dir" \
    --ir-dir "$ir_dir" \
    --vis-dir "$vis_dir" \
    --checkpoint-dir "$checkpoint_dir" \
    --epoch "$epoch"

# 检查执行结果
if [ $? -eq 0 ]; then
    echo ""
    echo "✅ MSRS 图像融合测试完成!"
    echo "✅ 结果已保存到: $output_dir"
else
    echo ""
    echo "❌ MSRS 图像融合测试失败!"
    exit 1
fi