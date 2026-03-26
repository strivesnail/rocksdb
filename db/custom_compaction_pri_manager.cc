//  Copyright (c) 2011-present, Facebook, Inc.  All rights reserved.
//  This source code is licensed under both the GPLv2 (found in the
//  COPYING file in the root directory) and Apache 2.0 License
//  (found in the LICENSE.Apache file in the root directory).

#include "db/custom_compaction_pri_manager.h"

#include <atomic>
#include <cinttypes>
#include <cstdlib>
#include <cstring>
#include <mutex>

#include "rocksdb/env.h"
#include "logging/logging.h"

namespace ROCKSDB_NAMESPACE {

// 全局单例
CustomCompactionPriManager* g_custom_compaction_pri_manager = nullptr;

CustomCompactionPriManager::CustomCompactionPriManager()
    : num_levels_(0), info_log_(nullptr) {
}

CustomCompactionPriManager::~CustomCompactionPriManager() {
}

void CustomCompactionPriManager::Initialize(int num_levels, Logger* info_log) {
  std::unique_lock<std::shared_mutex> lock(rw_mutex_);
  num_levels_ = num_levels;
  info_log_ = info_log;
  // 使用unique_ptr避免std::atomic的复制/移动问题
  level_states_.clear();
  level_states_.reserve(num_levels_);
  for (int i = 0; i < num_levels_; i++) {
    level_states_.emplace_back(std::make_unique<LevelCompactionPriState>());
    level_states_[i]->compaction_count.store(0, std::memory_order_relaxed);
  }
  initialized_.store(true, std::memory_order_release);
}

bool CustomCompactionPriManager::IsEnabled() {
  // 显式关闭：ROCKSDB_ENABLE_CUSTOM_COMPACTION=0 时不再走自定义 compaction
  const char* env_custom = std::getenv("ROCKSDB_ENABLE_CUSTOM_COMPACTION");
  if (env_custom != nullptr && std::string(env_custom) == "0") {
    return false;
  }
  // 由环境变量 ROCKSDB_ENABLE_PREDICTED_COMPACTION=1 或 ROCKSDB_ENABLE_PHASE2=1 控制
  const char* env_pred = std::getenv("ROCKSDB_ENABLE_PREDICTED_COMPACTION");
  const char* env_phase2 = std::getenv("ROCKSDB_ENABLE_PHASE2");
  return (env_pred != nullptr && std::string(env_pred) == "1") ||
         (env_phase2 != nullptr && std::string(env_phase2) == "1");
}

int CustomCompactionPriManager::GetCompactionRatio() {
  // 从环境变量读取比例，格式：ROCKSDB_COMPACTION_PRI_RATIO=3:1 或 1:1
  // 默认1:1（ratio=2）；任何回退均打日志（各原因至多一次，避免刷屏）
  auto log_ratio_fallback = [](const char* reason) {
    Logger* lg = g_custom_compaction_pri_manager
                     ? g_custom_compaction_pri_manager->info_log_
                     : nullptr;
    if (!lg) {
      return;
    }
    ROCKS_LOG_WARN(lg,
                   "[FALLBACK][CustomCompactionPri] "
                   "ROCKSDB_COMPACTION_PRI_RATIO %s; using default 1:1 "
                   "(ratio sum=2).",
                   reason);
  };

  const char* env_val = std::getenv("ROCKSDB_COMPACTION_PRI_RATIO");
  if (env_val == nullptr) {
    static std::atomic<bool> logged_unset{false};
    if (!logged_unset.exchange(true)) {
      log_ratio_fallback("unset");
    }
    return 2;  // 默认1:1
  }
  
  std::string ratio_str(env_val);
  // 解析格式 "3:1" 或 "1:1"
  size_t colon_pos = ratio_str.find(':');
  if (colon_pos == std::string::npos) {
    static std::atomic<bool> logged_fmt{false};
    if (!logged_fmt.exchange(true)) {
      log_ratio_fallback("invalid format (expected A:B)");
    }
    return 2;  // 格式错误，使用默认1:1
  }
  
  try {
    int normal_count = std::stoi(ratio_str.substr(0, colon_pos));
    int custom_count = std::stoi(ratio_str.substr(colon_pos + 1));
    if (normal_count <= 0 || custom_count <= 0) {
      static std::atomic<bool> logged_bad{false};
      if (!logged_bad.exchange(true)) {
        log_ratio_fallback("non-positive components");
      }
      return 2;  // 无效值，使用默认1:1
    }
    // 返回总数，例如3:1返回4，1:1返回2
    return normal_count + custom_count;
  } catch (...) {
    static std::atomic<bool> logged_ex{false};
    if (!logged_ex.exchange(true)) {
      log_ratio_fallback("parse exception");
    }
    return 2;  // 解析失败，使用默认1:1
  }
}

bool CustomCompactionPriManager::ShouldUseCustomPri(int level) const {
  if (!IsEnabled()) {
    return false;
  }
  
  if (!initialized_.load(std::memory_order_acquire)) {
    return false;
  }
  
  if (level < 0 || level >= num_levels_) {
    ROCKS_LOG_WARN(info_log_, "[CustomCompactionPri] ShouldUseCustomPri(level=%d): invalid level (num_levels_=%d), returning false", 
            level, num_levels_);
    return false;
  }
  
  // 读锁保护
  std::shared_lock<std::shared_mutex> lock(rw_mutex_);
  
  // 双重检查：确保level_states_已经初始化且level在范围内
  if (level < 0 || level >= static_cast<int>(level_states_.size())) {
    ROCKS_LOG_WARN(info_log_, "[CustomCompactionPri] ShouldUseCustomPri(level=%d): level out of range (level_states_.size()=%zu), returning false", 
            level, level_states_.size());
    return false;
  }
  
  // 检查level_states_[level]是否为nullptr（防御性编程）
  if (level_states_[level] == nullptr) {
    ROCKS_LOG_ERROR(info_log_, "[CustomCompactionPri] ShouldUseCustomPri(level=%d): level_states_[%d] is nullptr, returning false", 
            level, level);
    return false;
  }
  
  // 获取该level的compaction计数（每个level独立计数）
  uint64_t count = level_states_[level]->compaction_count.load(std::memory_order_acquire);
  uint64_t ratio = static_cast<uint64_t>(GetCompactionRatio());
  
  uint64_t count_verify = level_states_[level]->compaction_count.load(std::memory_order_relaxed);
  if (count != count_verify) {
    ROCKS_LOG_ERROR(info_log_, "[CustomCompactionPri] ShouldUseCustomPri(level=%d): count mismatch! acquire=%lu, relaxed=%lu", 
            level, (unsigned long)count, (unsigned long)count_verify);
  }
  
  (void)ratio;
  return true;  // 始终使用自定义 pri
}

uint64_t CustomCompactionPriManager::GetCompactionCount(int level) const {
  if (!initialized_.load(std::memory_order_acquire) ||
      level < 0 || level >= num_levels_ ||
      level >= static_cast<int>(level_states_.size()) ||
      level_states_[level] == nullptr) {
    return 0;
  }
  return level_states_[level]->compaction_count.load(std::memory_order_acquire);
}

void CustomCompactionPriManager::IncrementCompactionCount(int level) {
  if (!IsEnabled()) {
    return;
  }
  
  if (!initialized_.load(std::memory_order_acquire)) {
    return;
  }
  
  if (level < 0 || level >= num_levels_) {
    return;
  }
  
  // 写锁保护
  std::unique_lock<std::shared_mutex> lock(rw_mutex_);
  
  // 双重检查：确保level_states_已经初始化且level在范围内
  if (level < 0 || level >= static_cast<int>(level_states_.size())) {
    return;
  }
  
  // 检查level_states_[level]是否为nullptr（防御性编程）
  if (level_states_[level] == nullptr) {
    return;
  }
  
  level_states_[level]->compaction_count.fetch_add(1, std::memory_order_release);
}

void CustomCompactionPriManager::Reset() {
  std::unique_lock<std::shared_mutex> lock(rw_mutex_);
  for (int i = 0; i < num_levels_; i++) {
    level_states_[i]->compaction_count.store(0, std::memory_order_relaxed);
  }
}

}  // namespace ROCKSDB_NAMESPACE

