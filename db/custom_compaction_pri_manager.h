//  Copyright (c) 2011-present, Facebook, Inc.  All rights reserved.
//  This source code is licensed under both the GPLv2 (found in the
//  COPYING file in the root directory) and Apache 2.0 License
//  (found in the LICENSE.Apache file in the root directory).

#pragma once

#include "rocksdb/rocksdb_namespace.h"

#include <atomic>
#include <memory>
#include <shared_mutex>
#include <vector>

namespace ROCKSDB_NAMESPACE {

// Forward declaration
class Logger;

// 缓存行大小（通常为64字节）
constexpr size_t kCacheLineSize = 64;

// 每个level的compaction优先级状态（缓存对齐）
struct alignas(kCacheLineSize) LevelCompactionPriState {
  // compaction计数器（每次compaction完成后+1）
  // 用于控制自定义pri和kMinOverlappingRatio的比例
  // 例如：1:1比例用 count % 2 == 0 判断，3:1比例用 count % 4 == 0 判断
  std::atomic<uint64_t> compaction_count{0};
  
  // 填充到缓存行大小，避免false sharing
  char padding[kCacheLineSize - sizeof(std::atomic<uint64_t>)];
};

static_assert(sizeof(LevelCompactionPriState) == kCacheLineSize,
              "LevelCompactionPriState must be cache-line aligned");

// 自定义Compaction优先级管理器
// 管理每个level的compaction优先级策略（自定义pri vs kMinOverlappingRatio）
class CustomCompactionPriManager {
 public:
  CustomCompactionPriManager();
  ~CustomCompactionPriManager();

  // 初始化（设置level数量）
  // info_log: RocksDB 日志对象，用于输出日志到 LOG 文件
  void Initialize(int num_levels, Logger* info_log = nullptr);

  // 检查是否启用自定义compaction优先级
  // 通过环境变量 ROCKSDB_ENABLE_PREDICTED_COMPACTION 控制
  static bool IsEnabled();

  // 获取指定level应该使用的compaction优先级策略
  // 返回true表示使用自定义pri，false表示使用kMinOverlappingRatio
  // 基于计数器和比例来判断：count % ratio == 0 时使用自定义pri
  bool ShouldUseCustomPri(int level) const;

  // 更新指定level的compaction计数器
  // 在compaction完成后调用，计数器+1
  void IncrementCompactionCount(int level);

  // 获取指定level的compaction计数（用于 CustomCompactionPicker 的 1:1 轮换）
  uint64_t GetCompactionCount(int level) const;

  // 获取compaction比例（正常compaction:自定义compaction）
  // 例如：1:1返回2，3:1返回4
  static int GetCompactionRatio();

  // 重置所有level的状态（计数器归零）
  void Reset();

 private:
  // 每个level的状态（缓存对齐，避免false sharing）
  // 使用unique_ptr数组避免std::atomic的复制/移动问题
  std::vector<std::unique_ptr<LevelCompactionPriState>> level_states_;
  
  // 读写锁保护level_states_的访问
  mutable std::shared_mutex rw_mutex_;
  
  // level数量
  int num_levels_;
  
  // 是否已初始化
  std::atomic<bool> initialized_{false};
  
  // RocksDB 日志对象，用于输出日志到 LOG 文件
  Logger* info_log_;
};

// 全局单例
extern CustomCompactionPriManager* g_custom_compaction_pri_manager;

}  // namespace ROCKSDB_NAMESPACE

