#!/bin/bash

# 设置最大重试次数
MAX_RETRIES=1000

# 初始化重试计数
retry_count=0

# 定义训练命令（替换为你的实际命令）
TRAIN_CMD="python3 basicsr/train.py --opt Options/QuadPriorFormer_LOL_v2_synthetic.yml"

# 添加信号处理：捕获Ctrl+C（SIGINT）并终止脚本
trap_ctrl_c() {
    echo -e "\n\n接收到终止信号，正在停止脚本..."
    exit 0
}
trap 'trap_ctrl_c' INT

while [ $retry_count -lt $MAX_RETRIES ]; do
    echo "开始第 $((retry_count + 1)) 次训练尝试..."
    $TRAIN_CMD
    
    # 检查训练是否成功（退出码为 0 表示成功）
    if [ $? -eq 0 ]; then
        echo "训练成功完成。"
        break
    else
        retry_count=$((retry_count + 1))
        echo "训练失败。正在重试... (第 $retry_count/$MAX_RETRIES 次)"
        sleep 3  # 等待 3 秒再重试（可选）
    fi
done

# 检查是否达到最大重试次数
if [ $retry_count -eq $MAX_RETRIES ]; then
    echo "达到最大重试次数，训练失败。"
fi
