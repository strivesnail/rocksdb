//  Copyright (c) 2011-present, Facebook, Inc.  All rights reserved.
//  This source code is licensed under both the GPLv2 (found in the
//  COPYING file in the root directory) and Apache 2.0 License
//  (found in the LICENSE.Apache file in the root directory).

#pragma once

#include <atomic>
#include <memory>
#include <shared_mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "rocksdb/env.h"
#include "rocksdb/status.h"
#include "rocksdb/slice.h"

namespace ROCKSDB_NAMESPACE {

// Forward declaration
class FSWritableFile;

// 文件元数据（仅用于两阶段写入，不管理删除）
struct FileMetadata {
  uint64_t file_number;
  int level;
  uint64_t creation_time;           // 创建时间（微秒）
  double predicted_lifetime;        // 预测的生命周期（秒）
  std::string target_file_path;     // 目标文件路径（db1/）
  int target_handle;                // 目标handle
  uint64_t file_size;               // 文件大小（字节）
};


// 两阶段写入管理器
class TwoPhaseWriteManager {
 public:
  TwoPhaseWriteManager();
  ~TwoPhaseWriteManager();

  // 初始化（加载模型等）
  // enable_phase2: 是否启用Phase 2的两阶段写入（只有Phase 2才需要预测和二次写入）
  Status Initialize(const std::string& model_dir, const std::string& db_path,
                   bool enable_phase2 = false);

  // 处理文件创建（第一阶段：写入内存）
  // 返回：是否应该写入内存（Level 0返回false，直接写入目标目录）
  bool HandleFileCreation(uint64_t file_number, int level, 
                          const std::string& file_path);
  
  // 创建内存缓冲区用于存储文件数据
  Status CreateMemoryBuffer(uint64_t file_number);
  
  // 将数据追加到内存缓冲区
  Status AppendToMemoryBuffer(uint64_t file_number, const Slice& data);
  
  // 创建内存文件写入器（用于compaction_job.cc）
  // 返回一个FSWritableFile实现，直接将数据写入内存缓冲区
  std::unique_ptr<FSWritableFile> CreateMemoryWritableFile(uint64_t file_number);
  
  // 完成内存缓冲区写入（标记为完成，可以用于重写）
  Status FinishMemoryBuffer(uint64_t file_number);
  
  // 从内存缓冲区获取数据（用于重写到目标文件）
  Status GetMemoryBufferData(uint64_t file_number, std::string* data);
  
  // 释放内存缓冲区
  void ReleaseMemoryBuffer(uint64_t file_number);

  // 处理特征收集和预测（文件创建后调用）
  Status CollectFeaturesAndPredict(uint64_t file_number, int level,
                                   const std::vector<double>& features);

  // 处理文件重新写入（第二阶段：根据预测结果写入目标handle）
  Status RewriteFileToTargetHandle(uint64_t file_number);

  // 关闭管理器
  void Shutdown();

  // 检查是否已初始化
  bool IsInitialized() const { return initialized_.load(); }

  // 检查是否启用Phase 2
  bool IsPhase2Enabled() const { return enable_phase2_; }

 private:
  // 模型加载
  Status LoadModels(const std::string& model_dir);

  // 预测生命周期
  double PredictLifetime(int level, const std::vector<double>& features);

  // 映射生命周期到handle
  int MapLifetimeToHandle(double predicted_lifetime_seconds, int level);
  
  // 获取handle的阈值（用于判断文件是否"too far"）
  // 返回该handle对应的最大生命周期（秒）
  double GetHandleThreshold(int handle) const;
  
  // 获取handle的上下界限
  // lower_bound: 该handle的最小生命周期（秒，包含）
  // upper_bound: 该handle的最大生命周期（秒，不包含）
  void GetHandleBounds(int handle, double* lower_bound, double* upper_bound) const;

  // 将内存缓冲区数据写入目标文件（带handle）
  Status WriteMemoryBufferToFile(uint64_t file_number,
                                 const std::string& buffer_data,
                                 const std::string& target_path,
                                 int target_handle);

  // 原子性移动文件（已废弃，保留用于兼容）
  Status MoveFileAtomically(const std::string& src_path, 
                           const std::string& dst_path, int target_handle);

  // 成员变量
  std::string db_path_;
  std::string model_dir_;
  bool enable_phase2_;  // 是否启用Phase 2的两阶段写入
  
  // 文件元数据映射（仅用于两阶段写入，不管理删除）
  std::unordered_map<uint64_t, FileMetadata> file_metadata_;
  std::shared_mutex metadata_mutex_;  // 读写锁：读操作（查找）可以并发，写操作（插入/删除/更新）需要独占
  
  // 内存缓冲区映射（存储文件数据）
  std::unordered_map<uint64_t, std::string> memory_buffers_;
  std::shared_mutex buffer_mutex_;  // 读写锁：保护内存缓冲区

  // 模型相关（需要根据实际模型库实现）
  // 这里使用void*作为占位符，实际需要根据使用的模型库（如LightGBM C API）定义
  void* models_[7];  // level 0-6的模型
  void* scalers_[7];  // level 0-6的scaler
  
  std::atomic<bool> initialized_;
};

// 全局单例（由DBImpl管理生命周期）
extern TwoPhaseWriteManager* g_two_phase_write_manager;

}  // namespace ROCKSDB_NAMESPACE

