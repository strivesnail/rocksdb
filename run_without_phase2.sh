#!/bin/bash
# ============================================
# RocksDB 运行脚本 - 禁用二次写入（Phase 1模式）
# ============================================

# 明确禁用二次写入
unset ROCKSDB_ENABLE_PHASE2

# 或者显式设置为0
# export ROCKSDB_ENABLE_PHASE2=0

# 显示配置信息
echo "============================================"
echo "RocksDB 运行配置"
echo "============================================"
echo "二次写入 (Phase 2): 禁用（Phase 1模式）"
echo "============================================"
echo ""

# 执行你的RocksDB程序
if [ $# -eq 0 ]; then
    echo "使用方法: $0 <your_rocksdb_program> [arguments...]"
    echo "示例: $0 ./db_bench --benchmarks=fillrandom"
    exit 1
fi

# 执行程序，传递所有参数
exec "$@"

