#!/bin/bash
# ============================================
# RocksDB 运行脚本 - 支持环境变量控制二次写入
# ============================================

# 设置是否启用二次写入（Phase 2）
# 设置为 1 启用，设置为 0 或注释掉则禁用
export ROCKSDB_ENABLE_PHASE2=1

# 设置ML模型目录路径
# 如果未设置，将使用代码中的默认路径
export ROCKSDB_ML_MODELS_PATH="/home/usr/test_environment/workplace/log/experiment_logs/new_log/run_batch_20260131_063807/backup_old_data_models/models_balanced"

# 设置Python工具路径（可选，用于ml_predict_lifetime_by_level.py）
# 如果未设置，将使用代码中的默认路径（./tools）
export ROCKSDB_TOOLS_PATH="/home/usr/test_environment/rocksdb/tools"

# 显示配置信息
echo "============================================"
echo "RocksDB 运行配置"
echo "============================================"
echo "二次写入 (Phase 2): $([ "$ROCKSDB_ENABLE_PHASE2" == "1" ] && echo "启用" || echo "禁用")"
echo "模型目录: ${ROCKSDB_ML_MODELS_PATH:-<默认路径>}"
echo "工具路径: ${ROCKSDB_TOOLS_PATH:-<默认路径>}"
echo "============================================"
echo ""

# 执行你的RocksDB程序
# 替换下面的命令为你的实际程序
if [ $# -eq 0 ]; then
    echo "使用方法: $0 <your_rocksdb_program> [arguments...]"
    echo "示例: $0 ./db_bench --benchmarks=fillrandom"
    exit 1
fi

# 执行程序，传递所有参数
exec "$@"

