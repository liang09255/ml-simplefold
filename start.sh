#!/bin/bash
# start.sh — 一键恢复 SimpleFold 训练环境
# 在你本机每次重新打开终端时运行

set -e

cd /root/code/ml-simplefold

# 1. 激活虚拟环境
echo "[1/4] Activating virtual environment..."
source venv/bin/activate

# 2. 检查 Redis 是否运行，没跑则启动
echo "[2/4] Checking Redis..."
if redis-cli -p 7777 ping 2>/dev/null | grep -q PONG; then
    echo "  Redis is already running."
else
    echo "  Starting Redis with CCD dictionary..."
    redis-server --dbfilename ccd.rdb --port 7777 --daemonize yes
    sleep 1
    echo "  Redis started."
fi

# 3. 检查 CCD 数据
echo "[3/4] Checking CCD data..."
ENTRIES=$(redis-cli -p 7777 --raw dbsize)
echo "  CCD has $ENTRIES entries."

# 4. 显示训练数据状态
echo "[4/4] Checking training data..."
STRUCTURES=$(ls data/processed_targets/structures/*.npz 2>/dev/null | wc -l)
TOKENS=$(ls data/tokenized/tokens/*.pkl 2>/dev/null | wc -l)
echo "  $STRUCTURES processed structures, $TOKENS tokenized samples."

echo ""
echo "=== Environment ready ==="
echo "Run training:  python src/simplefold/train.py experiment=debug_cpu"
echo "Run tensorboard: tensorboard --logdir artifacts/tensorboard --port 6006"
echo ""
