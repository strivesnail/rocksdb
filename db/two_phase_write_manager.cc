//  Copyright (c) 2011-present, Facebook, Inc.  All rights reserved.
//  This source code is licensed under both the GPLv2 (found in the
//  COPYING file in the root directory) and Apache 2.0 License
//  (found in the LICENSE.Apache file in the root directory).

#include "db/two_phase_write_manager.h"

#include "rocksdb/io_status.h"

#include <algorithm>
#include <cmath>
#include <cinttypes>
#include <cstring>
#include <fstream>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <fcntl.h>
#include <linux/fs.h>
#include <errno.h>
#include <stdlib.h>  // for posix_memalign
#include <thread>    // for std::this_thread::sleep_for
#include <chrono>    // for std::chrono::milliseconds

#include "rocksdb/env.h"
#include "rocksdb/status.h"
#include "rocksdb/slice.h"
#include "rocksdb/file_system.h"
#include "util/string_util.h"
#include "db/event_helpers_ml_features.h"
#include "tools/ml_predict_python.h"
#include "file/writable_file_writer.h"
#include "file/sequence_file_reader.h"
#include "file/file_util.h"
#include "file/filename.h"
#include "monitoring/histogram.h"

namespace ROCKSDB_NAMESPACE {

// 全局单例（由DBImpl管理生命周期）
TwoPhaseWriteManager* g_two_phase_write_manager = nullptr;

// FDP Hint定义
#define RWH_WRITE_LIFE_NOT_SET   0
#define RWH_WRITE_LIFE_NONE      1
#define RWH_WRITE_LIFE_SHORT     2
#define RWH_WRITE_LIFE_MEDIUM    3
#define RWH_WRITE_LIFE_LONG      4
#define RWH_WRITE_LIFE_EXTREME   5

TwoPhaseWriteManager::TwoPhaseWriteManager()
    : enable_phase2_(false),
      initialized_(false) {
  memset(models_, 0, sizeof(models_));
  memset(scalers_, 0, sizeof(scalers_));
}

TwoPhaseWriteManager::~TwoPhaseWriteManager() {
  Shutdown();
}

Status TwoPhaseWriteManager::Initialize(const std::string& model_dir,
                                        const std::string& db_path,
                                        bool enable_phase2) {
  fprintf(stderr, "[TwoPhaseWriteManager::Initialize] [VERIFY] Starting initialization...\n");
  fprintf(stderr, "[TwoPhaseWriteManager::Initialize] [VERIFY]   model_dir=%s\n", model_dir.c_str());
  fprintf(stderr, "[TwoPhaseWriteManager::Initialize] [VERIFY]   db_path=%s\n", db_path.c_str());
  fprintf(stderr, "[TwoPhaseWriteManager::Initialize] [VERIFY]   enable_phase2=%d\n", enable_phase2 ? 1 : 0);
  fflush(stderr);
  
  if (initialized_.load()) {
    return Status::OK();
  }

  model_dir_ = model_dir;
  db_path_ = db_path;
  enable_phase2_ = enable_phase2;

  
  // 打印所有handle的上下界限（用于调试和验证）
  if (enable_phase2_) {
    fprintf(stderr, "[HANDLE_BOUNDS] ================================================================\n");
    fprintf(stderr, "[HANDLE_BOUNDS] Handle阈值定义（用于根据预测结果选择handle和判断文件是否too far）:\n");
    for (int h = 6; h <= 12; h++) {
      double lower, upper;
      GetHandleBounds(h, &lower, &upper);
      fprintf(stderr, "[HANDLE_BOUNDS]   Handle %d: [%.1f, %.1f) seconds, threshold=%.1f seconds\n",
              h, lower, upper, GetHandleThreshold(h));
    }
    fprintf(stderr, "[HANDLE_BOUNDS] ================================================================\n");
    fflush(stderr);
  }

  // 只有Phase 2才需要加载模型和启动后台线程
  if (enable_phase2_) {
    fprintf(stderr, "[TwoPhaseWriteManager] Loading models from: %s\n", model_dir.c_str());
    fflush(stderr);
    fflush(stderr);
    
    // 加载模型 - NO ERROR TOLERANCE
    Status s = LoadModels(model_dir);
    if (!s.ok()) {
      fprintf(stderr, "[TwoPhaseWriteManager::Initialize] [FATAL] Failed to load models from %s: %s\n",
              model_dir.c_str(), s.ToString().c_str());
      fflush(stderr);
      return s;  // FAIL IMMEDIATELY
    }
    
    fprintf(stderr, "[TwoPhaseWriteManager] ✓ Models loaded successfully\n");
    fflush(stderr);
  } else {
  }

  initialized_.store(true);
  fprintf(stderr, "[TwoPhaseWriteManager] ✓ Initialization completed (phase2=%d)\n", enable_phase2 ? 1 : 0);
  fflush(stderr);
  return Status::OK();
}

bool TwoPhaseWriteManager::HandleFileCreation(uint64_t file_number, int level,
                                              const std::string& file_path) {
  // Phase 1不需要两阶段写入，直接写入目标目录
  if (!enable_phase2_) {
    return false;  // 返回false表示不写内存，直接写入目标目录
  }

  // NO ERROR TOLERANCE - 如果未初始化，这是严重错误
  if (!initialized_.load()) {
    fprintf(stderr, "[FATAL] HandleFileCreation: file #%" PRIu64 " - 未初始化！这是严重错误！\n", file_number);
    fflush(stderr);
    abort();  // FAIL IMMEDIATELY - NO FALLBACK
  }

  // Level 0文件直接写入目标目录（使用handle6），不写内存
  if (level == 0) {
    return false;  // 返回false表示不写内存
  }

  // Phase 2的其他文件先写内存缓冲区
  
  // 创建内存缓冲区
  Status s = CreateMemoryBuffer(file_number);
  if (!s.ok()) {
    fprintf(stderr, "[FATAL] HandleFileCreation: file #%" PRIu64 " - 创建内存缓冲区失败: %s\n",
            file_number, s.ToString().c_str());
    fflush(stderr);
    abort();  // FAIL IMMEDIATELY
  }
  
  return true;  // 返回true表示写内存缓冲区
}

Status TwoPhaseWriteManager::CollectFeaturesAndPredict(
    uint64_t file_number, int level, const std::vector<double>& features) {
  // Phase 1不需要预测
  if (!enable_phase2_) {
    return Status::OK();  // Phase 1直接返回OK，不进行预测
  }

  if (!initialized_.load() || level <= 0 || level > 6) {
    return Status::InvalidArgument("Invalid level or not initialized");
  }

  if (features.size() != 69) {
    return Status::InvalidArgument("Invalid feature count");
  }

  // 性能测量：ML预测
  
  uint64_t t0_predict = Env::Default()->NowMicros();
  double predicted_lifetime = PredictLifetime(level, features);
  uint64_t t1_predict = Env::Default()->NowMicros();
  // 性能日志已简化
  
  
  if (predicted_lifetime < 0) {
    fprintf(stderr, "[ERROR] CollectFeaturesAndPredict: file #%" PRIu64 " - ✗ 预测失败！详细信息：\n",
            file_number);
    fprintf(stderr, "  - predicted_lifetime: %.6f (负数表示失败)\n", predicted_lifetime);
    fprintf(stderr, "  - level: %d\n", level);
    fprintf(stderr, "  - features.size(): %zu\n", features.size());
    fprintf(stderr, "  - enable_phase2_: %d\n", enable_phase2_ ? 1 : 0);
    fprintf(stderr, "  - initialized_: %d\n", initialized_.load() ? 1 : 0);
    
    // 打印前几个特征值用于调试
    if (features.size() > 0) {
      fprintf(stderr, "  - 前5个特征值: ");
      for (size_t i = 0; i < std::min(features.size(), size_t(5)); i++) {
        fprintf(stderr, "%.6f ", features[i]);
      }
      fprintf(stderr, "\n");
    }
    
    return Status::Corruption("Prediction failed: PredictLifetime returned negative value " + 
                             std::to_string(predicted_lifetime) + " for file #" + 
                             std::to_string(file_number));
  }

  // 获取当前时间
  uint64_t current_time = Env::Default()->NowMicros();

  // 根据预测结果选择handle
  int target_handle = MapLifetimeToHandle(predicted_lifetime, level);
  
  // 简化的handle选择日志
  fprintf(stderr, "[HANDLE_SELECT] file #%" PRIu64 " - handle=%d, lifetime=%.2fs\n", 
          file_number, target_handle, predicted_lifetime);
  fflush(stderr);

  // 构建文件路径（使用RocksDB的文件名格式，带前导零）
  // 格式：000012.sst（6位数字，不足补0）
  char target_filename[32];
  snprintf(target_filename, sizeof(target_filename), "%06" PRIu64 ".sst", file_number);
  std::string target_file_path = db_path_ + "/" + target_filename;
  
  // 获取内存缓冲区大小
  std::string buffer_data;
  Status buffer_status = GetMemoryBufferData(file_number, &buffer_data);
  uint64_t file_size = 0;
  if (buffer_status.ok()) {
    file_size = buffer_data.size();
  }

  // 保存元数据（写操作，使用unique_lock）
  {
    std::unique_lock<std::shared_mutex> lock(metadata_mutex_);
    file_metadata_[file_number] = {
        file_number,
        level,
        current_time,
        predicted_lifetime,
        target_file_path,
        target_handle,
        file_size
    };
  }

  // 预测成功，已记录在HANDLE_SELECT日志中

  return Status::OK();
}

Status TwoPhaseWriteManager::RewriteFileToTargetHandle(uint64_t file_number) {
  
  // Phase 1不需要重新写入
  if (!enable_phase2_) {
    return Status::OK();  // Phase 1直接返回OK，不进行重新写入
  }

  // NO ERROR TOLERANCE - 如果未初始化，这是严重错误
  if (!initialized_.load()) {
    fprintf(stderr, "[FATAL] RewriteFileToTargetHandle: file #%" PRIu64 " - 未初始化！这是严重错误！\n", file_number);
    fflush(stderr);
    abort();  // FAIL IMMEDIATELY - NO FALLBACK
  }

  FileMetadata metadata;
  {
    // 读操作，使用shared_lock
    std::shared_lock<std::shared_mutex> lock(metadata_mutex_);
    auto it = file_metadata_.find(file_number);
    if (it == file_metadata_.end()) {
      // NO ERROR TOLERANCE - File metadata not found is a fatal error
      fprintf(stderr, "[FATAL] RewriteFileToTargetHandle: file #%" PRIu64 " - File metadata not found! This is a fatal error!\n", file_number);
      fflush(stderr);
      abort();  // FAIL IMMEDIATELY
    }
    metadata = it->second;
  }

  // Level 0文件已经在目标目录，不需要重新写入
  if (metadata.level == 0) {
    return Status::OK();
  }

  // 构建正确的目标路径（使用RocksDB的文件名格式，带前导零）
  // 格式：000012.sst（6位数字，不足补0）
  char target_filename[32];
  snprintf(target_filename, sizeof(target_filename), "%06" PRIu64 ".sst", file_number);
  std::string correct_target_path = db_path_ + "/" + target_filename;
  
  // 二次写入开始，日志已简化
  
  // 从内存缓冲区获取数据
  std::string buffer_data;
  Status s = GetMemoryBufferData(file_number, &buffer_data);
  if (!s.ok()) {
    fprintf(stderr, "[FATAL] RewriteFileToTargetHandle: file #%" PRIu64 " - 获取内存缓冲区数据失败: %s\n",
            file_number, s.ToString().c_str());
    fflush(stderr);
    abort();  // FAIL IMMEDIATELY
  }
  
  
  // 从内存缓冲区写入目标文件（使用正确的handle） - NO ERROR TOLERANCE
  s = WriteMemoryBufferToFile(file_number, buffer_data, correct_target_path, metadata.target_handle);
  if (!s.ok()) {
    fprintf(stderr, "[FATAL] RewriteFileToTargetHandle: file #%" PRIu64 " - 写入目标文件失败: %s\n", 
            file_number, s.ToString().c_str());
    fprintf(stderr, "[FATAL] RewriteFileToTargetHandle: target_path=%s\n", correct_target_path.c_str());
    fprintf(stderr, "[FATAL] RewriteFileToTargetHandle: handle=%d\n", metadata.target_handle);
    fflush(stderr);
    abort();  // FAIL IMMEDIATELY - NO FALLBACK
  }
  
  // 释放内存缓冲区
  ReleaseMemoryBuffer(file_number);
  
  // 二次写入成功，日志已简化

  return Status::OK();
}

// Handle阈值定义（秒）
// 每个handle对应一个生命周期范围：[lower_bound, upper_bound)
namespace {
  constexpr double kHandle6Lower = 0.0;
  constexpr double kHandle6Upper = 8.0;
  constexpr double kHandle7Lower = 8.0;
  constexpr double kHandle7Upper = 12.0;
  constexpr double kHandle8Lower = 12.0;
  constexpr double kHandle8Upper = 30.0;
  constexpr double kHandle9Lower = 30.0;
  constexpr double kHandle9Upper = 100.0;
  constexpr double kHandle10Lower = 100.0;
  constexpr double kHandle10Upper = 300.0;
  constexpr double kHandle11Lower = 300.0;
  constexpr double kHandle11Upper = 600.0;
  constexpr double kHandle12Lower = 600.0;
  constexpr double kHandle12Upper = std::numeric_limits<double>::max();
}

int TwoPhaseWriteManager::MapLifetimeToHandle(double predicted_lifetime_seconds,
                                              int level) {
  // Level 0文件直接返回handle6
  if (level == 0) {
    // Level 0总是使用handle 6
    return 6;
  }

  // 根据预测的生命周期映射到handle
  int target_handle;
  double lower_bound, upper_bound;
  
  if (predicted_lifetime_seconds < kHandle6Upper) {
    target_handle = 6;
    lower_bound = kHandle6Lower;
    upper_bound = kHandle6Upper;
  } else if (predicted_lifetime_seconds < kHandle7Upper) {
    target_handle = 7;
    lower_bound = kHandle7Lower;
    upper_bound = kHandle7Upper;
  } else if (predicted_lifetime_seconds < kHandle8Upper) {
    target_handle = 8;
    lower_bound = kHandle8Lower;
    upper_bound = kHandle8Upper;
  } else if (predicted_lifetime_seconds < kHandle9Upper) {
    target_handle = 9;
    lower_bound = kHandle9Lower;
    upper_bound = kHandle9Upper;
  } else if (predicted_lifetime_seconds < kHandle10Upper) {
    target_handle = 10;
    lower_bound = kHandle10Lower;
    upper_bound = kHandle10Upper;
  } else if (predicted_lifetime_seconds < kHandle11Upper) {
    target_handle = 11;
    lower_bound = kHandle11Lower;
    upper_bound = kHandle11Upper;
  } else {
    target_handle = 12;
    lower_bound = kHandle12Lower;
    upper_bound = kHandle12Upper;
  }
  
  // Handle映射日志已移除，信息在HANDLE_SELECT中
  
  return target_handle;
}

double TwoPhaseWriteManager::GetHandleThreshold(int handle) const {
  // 返回该handle对应的最大生命周期（用于判断文件是否"too far"）
  switch (handle) {
    case 6: return kHandle6Upper;
    case 7: return kHandle7Upper;
    case 8: return kHandle8Upper;
    case 9: return kHandle9Upper;
    case 10: return kHandle10Upper;
    case 11: return kHandle11Upper;
    case 12: return kHandle12Upper;
    default:
      fprintf(stderr, "[WARN] GetHandleThreshold: invalid handle %d, using handle 6 threshold\n", handle);
      fflush(stderr);
      return kHandle6Upper;
  }
}

void TwoPhaseWriteManager::GetHandleBounds(int handle, double* lower_bound, double* upper_bound) const {
  if (lower_bound == nullptr || upper_bound == nullptr) {
    return;
  }
  
  switch (handle) {
    case 6:
      *lower_bound = kHandle6Lower;
      *upper_bound = kHandle6Upper;
      break;
    case 7:
      *lower_bound = kHandle7Lower;
      *upper_bound = kHandle7Upper;
      break;
    case 8:
      *lower_bound = kHandle8Lower;
      *upper_bound = kHandle8Upper;
      break;
    case 9:
      *lower_bound = kHandle9Lower;
      *upper_bound = kHandle9Upper;
      break;
    case 10:
      *lower_bound = kHandle10Lower;
      *upper_bound = kHandle10Upper;
      break;
    case 11:
      *lower_bound = kHandle11Lower;
      *upper_bound = kHandle11Upper;
      break;
    case 12:
      *lower_bound = kHandle12Lower;
      *upper_bound = kHandle12Upper;
      break;
    default:
      fprintf(stderr, "[WARN] GetHandleBounds: invalid handle %d, using handle 6 bounds\n", handle);
      fflush(stderr);
      *lower_bound = kHandle6Lower;
      *upper_bound = kHandle6Upper;
      break;
  }
}

Status TwoPhaseWriteManager::MoveFileAtomically(const std::string& /*src_path*/,
                                                const std::string& /*dst_path*/,
                                                int /*target_handle*/) {
  // 已废弃：不再使用临时文件，改用内存缓冲区
  // 保留此函数用于兼容，但不应被调用
  fprintf(stderr, "[ERROR] MoveFileAtomically: 此函数已废弃，不应被调用！\n");
  fflush(stderr);
  return Status::NotSupported("MoveFileAtomically is deprecated, use WriteMemoryBufferToFile instead");
}

Status TwoPhaseWriteManager::WriteMemoryBufferToFile(uint64_t /*file_number*/,
                                                      const std::string& buffer_data,
                                                      const std::string& target_path,
                                                      int target_handle) {
  // 使用 RocksDB 的 WritableFileWriter 将内存缓冲区数据写入目标文件
  
  Env* env = Env::Default();
  FileSystem* fs = env->GetFileSystem().get();
  
  uint64_t buffer_size = buffer_data.size();
  
  // 性能测量：创建目标文件
  uint64_t t0_create_dst = Env::Default()->NowMicros();
  // 创建目标文件（使用和第一次写入相同的 FileOptions）
  // 注意：目标文件不应该已存在（file_number是唯一的），不需要删除
  FileOptions dst_options;
  dst_options.write_hint = static_cast<Env::WriteLifeTimeHint>(target_handle);
  
  std::unique_ptr<FSWritableFile> dst_file;
  IOStatus io_s = fs->NewWritableFile(target_path, dst_options, &dst_file, nullptr);
  if (!io_s.ok()) {
    return io_s;
  }
  
  // 设置 write hint（FDP handle）
  dst_file->SetWriteLifeTimeHint(dst_options.write_hint);
  uint64_t t1_create_dst = Env::Default()->NowMicros();
  fprintf(stderr, "[PERF] WriteMemoryBufferToFile - 创建目标文件耗时: %.3f ms\n",
          (t1_create_dst - t0_create_dst) / 1000.0);
  
  // 创建 WritableFileWriter（和第一次写入完全一样）
  std::unique_ptr<WritableFileWriter> writer(
      new WritableFileWriter(std::move(dst_file), target_path, dst_options,
                            nullptr, nullptr, nullptr, Histograms::SST_WRITE_MICROS,
                            std::vector<std::shared_ptr<EventListener>>(),
                            nullptr, false, false));
  
  // 性能测量：写入数据
  uint64_t t0_write = Env::Default()->NowMicros();
  // 将内存缓冲区数据写入文件
  Slice data_slice(buffer_data.data(), buffer_data.size());
  IOOptions io_opts;
  io_s = writer->Append(io_opts, data_slice);
  if (!io_s.ok()) {
    return io_s;
  }
  uint64_t t1_write = Env::Default()->NowMicros();
  // 性能日志已简化
  
  // 性能测量：关闭文件
  uint64_t t0_close = Env::Default()->NowMicros();
  // 关闭文件（确保文件完全写入并可见）
  // 注意：不调用Sync，Close会确保数据写入到文件系统
  io_s = writer->Close(io_opts);
  if (!io_s.ok()) {
    return io_s;
  }
  uint64_t t1_close = Env::Default()->NowMicros();
  // 性能日志已简化

  return Status::OK();
}

// 已移除：所有删除队列和移动队列相关函数 - 不再管理文件删除

Status TwoPhaseWriteManager::LoadModels(const std::string& model_dir) {
  (void)model_dir;  // 在函数内部使用，不需要标记为未使用
#ifdef ROCKSDB_ML_PREDICT_PYTHON
  // 使用Python API加载模型
  // 设置环境变量，让Python脚本知道模型路径
  setenv("ROCKSDB_ML_MODELS_PATH", model_dir.c_str(), 1);
  
  // 初始化Python预测器（如果还没有初始化）
  if (!InitializeMLPredictorByLevel()) {
    fprintf(stderr, "[TwoPhaseWriteManager::LoadModels] ERROR: InitializeMLPredictorByLevel() returned false\n");
    fflush(stderr);
    return Status::Corruption("Failed to initialize Python ML predictor");
  }
  
  return Status::OK();
#else
  fprintf(stderr, "[TwoPhaseWriteManager::LoadModels] ERROR: ROCKSDB_ML_PREDICT_PYTHON is NOT defined - Python ML prediction not enabled\n");
  fflush(stderr);
  return Status::NotSupported("Python ML prediction not enabled");
#endif
}

double TwoPhaseWriteManager::PredictLifetime(int level,
                                            const std::vector<double>& features) {
  fprintf(stderr, "[VERIFY] PredictLifetime: level=%d, features.size()=%zu\n",
          level, features.size());
  
#ifdef ROCKSDB_ML_PREDICT_PYTHON
  if (features.size() != 69) {
    fprintf(stderr, "[ERROR] PredictLifetime: 特征数量错误！期望69，实际%zu\n",
            features.size());
    return -1.0;  // 特征数量不对
  }
  
  fprintf(stderr, "[VERIFY] PredictLifetime: 调用PredictFileLifetimePythonByLevel\n");
  fflush(stderr);
  
  fprintf(stderr, "[DEBUG] PredictLifetime: 准备调用函数，features.data()=%p, features.size()=%zu, level=%d\n", 
          (void*)features.data(), features.size(), level);
  fflush(stderr);
  
  // 使用现有的预测函数
  double result = PredictFileLifetimePythonByLevel(features.data(), features.size(), level);
  
  
  fprintf(stderr, "[VERIFY] PredictLifetime: PredictFileLifetimePythonByLevel返回: %.6f\n",
          result);
  
  if (result < 0) {
    fprintf(stderr, "[FATAL] PredictLifetime: 预测函数返回负数！这表示预测失败。\n");
    fprintf(stderr, "  - level: %d\n", level);
    fprintf(stderr, "  - features.size(): %zu\n", features.size());
    fprintf(stderr, "  - 返回值: %.6f\n", result);
    fprintf(stderr, "[FATAL] 预测失败，程序将终止（NO ERROR TOLERANCE）\n");
    fflush(stderr);
    abort();  // NO ERROR TOLERANCE - 预测失败必须立即终止
    return -1.0;  // 不会执行到这里
  }
  
  // 检查特征值是否有效（在预测成功的情况下）
  bool has_invalid = false;
  for (size_t i = 0; i < features.size(); i++) {
    if (std::isnan(features[i]) || std::isinf(features[i])) {
      fprintf(stderr, "  - 特征[%zu]无效: %.6f\n", i, features[i]);
      has_invalid = true;
    }
  }
  if (has_invalid) {
    fprintf(stderr, "  - 发现无效特征值（NaN或Inf）\n");
  }
  
  return result;
#else
  fprintf(stderr, "[ERROR] PredictLifetime: ROCKSDB_ML_PREDICT_PYTHON未定义，不支持预测\n");
  return -1.0;  // 不支持
#endif
}

Status TwoPhaseWriteManager::CreateMemoryBuffer(uint64_t file_number) {
  std::unique_lock<std::shared_mutex> lock(buffer_mutex_);
  
  // 检查缓冲区是否已存在 - 如果存在，这是不应该发生的情况
  if (memory_buffers_.find(file_number) != memory_buffers_.end()) {
    fprintf(stderr, "[FATAL] CreateMemoryBuffer: file #%" PRIu64 " - 缓冲区已存在！这是不应该发生的情况！\n", file_number);
    fflush(stderr);
    abort();  // FAIL IMMEDIATELY - NO ERROR TOLERANCE
  }
  
  memory_buffers_[file_number] = std::string();
  memory_buffers_[file_number].reserve(64 * 1024 * 1024);  // 预分配64MB
  // 内存缓冲区已创建
  return Status::OK();
}

Status TwoPhaseWriteManager::AppendToMemoryBuffer(uint64_t file_number, const Slice& data) {
  std::unique_lock<std::shared_mutex> lock(buffer_mutex_);
  auto it = memory_buffers_.find(file_number);
  if (it == memory_buffers_.end()) {
    // 内存缓冲区未找到
    fflush(stderr);
    return Status::NotFound("Memory buffer not found for file " + std::to_string(file_number));
  }
  it->second.append(data.data(), data.size());
  return Status::OK();
}

Status TwoPhaseWriteManager::FinishMemoryBuffer(uint64_t /*file_number*/) {
  // 内存缓冲区已经完成，不需要额外操作
  // 这个方法保留用于未来可能的扩展
  return Status::OK();
}

Status TwoPhaseWriteManager::GetMemoryBufferData(uint64_t file_number, std::string* data) {
  std::shared_lock<std::shared_mutex> lock(buffer_mutex_);
  fprintf(stderr, "[DEBUG] GetMemoryBufferData: file #%" PRIu64 ", memory_buffers_.size()=%zu\n", 
          file_number, memory_buffers_.size());
  fflush(stderr);
  
  // Debug: print all file numbers in memory_buffers_
  if (memory_buffers_.size() > 0) {
    fprintf(stderr, "[DEBUG] GetMemoryBufferData: Available file numbers: ");
    for (const auto& pair : memory_buffers_) {
      fprintf(stderr, "%" PRIu64 " ", pair.first);
    }
    fprintf(stderr, "\n");
    fflush(stderr);
  }
  
  auto it = memory_buffers_.find(file_number);
  if (it == memory_buffers_.end()) {
    fprintf(stderr, "[DEBUG] GetMemoryBufferData: file #%" PRIu64 " NOT FOUND in memory_buffers_\n", file_number);
    fflush(stderr);
    return Status::NotFound("Memory buffer not found for file " + std::to_string(file_number));
  }
  *data = it->second;
  fprintf(stderr, "[DEBUG] GetMemoryBufferData: file #%" PRIu64 " found, size=%zu bytes\n", 
          file_number, it->second.size());
  fflush(stderr);
  return Status::OK();
}

void TwoPhaseWriteManager::ReleaseMemoryBuffer(uint64_t file_number) {
  std::unique_lock<std::shared_mutex> lock(buffer_mutex_);
  memory_buffers_.erase(file_number);
}

// MemoryWritableFile: 一个FSWritableFile实现，将数据写入内存缓冲区
namespace {
class MemoryWritableFile : public FSWritableFile {
 private:
  TwoPhaseWriteManager* manager_;
  uint64_t file_number_;
  std::string filename_;
  uint64_t offset_;

 public:
  MemoryWritableFile(TwoPhaseWriteManager* manager, uint64_t file_number, const std::string& filename)
      : manager_(manager), file_number_(file_number), filename_(filename), offset_(0) {}
  
  ~MemoryWritableFile() override {}
  
  IOStatus Append(const Slice& data, const IOOptions& /*options*/, IODebugContext* /*dbg*/) override {
    Status s = manager_->AppendToMemoryBuffer(file_number_, data);
    if (s.ok()) {
      offset_ += data.size();
      return IOStatus::OK();
    }
    return status_to_io_status(std::move(s));
  }
  
  IOStatus Append(const Slice& data, const IOOptions& options,
                  const DataVerificationInfo& /*verification_info*/,
                  IODebugContext* dbg) override {
    return Append(data, options, dbg);
  }
  
  IOStatus Close(const IOOptions& /*options*/, IODebugContext* /*dbg*/) override {
    Status s = manager_->FinishMemoryBuffer(file_number_);
    if (s.ok()) {
      return IOStatus::OK();
    }
    return status_to_io_status(std::move(s));
  }
  
  IOStatus Flush(const IOOptions& /*options*/, IODebugContext* /*dbg*/) override {
    return IOStatus::OK();  // 内存缓冲区不需要flush
  }
  
  IOStatus Sync(const IOOptions& /*options*/, IODebugContext* /*dbg*/) override {
    return IOStatus::OK();  // 内存缓冲区不需要sync
  }
  
  bool use_direct_io() const override { return false; }
  size_t GetRequiredBufferAlignment() const override { return 4096; }
  uint64_t GetFileSize(const IOOptions& /*options*/, IODebugContext* /*dbg*/) override {
    return offset_;
  }
  
  // 其他方法实现为NotSupported或no-op
  IOStatus PositionedAppend(const Slice& /*data*/, uint64_t /*offset*/, const IOOptions& /*options*/, IODebugContext* /*dbg*/) override {
    return IOStatus::NotSupported("PositionedAppend not supported for MemoryWritableFile");
  }
  
  IOStatus PositionedAppend(const Slice& data, uint64_t offset,
                           const IOOptions& options,
                           const DataVerificationInfo& /*verification_info*/,
                           IODebugContext* dbg) override {
    return PositionedAppend(data, offset, options, dbg);
  }
  
  IOStatus Truncate(uint64_t /*size*/, const IOOptions& /*options*/, IODebugContext* /*dbg*/) override {
    return IOStatus::NotSupported("Truncate not supported for MemoryWritableFile");
  }
  
  IOStatus RangeSync(uint64_t /*offset*/, uint64_t /*nbytes*/, const IOOptions& /*options*/, IODebugContext* /*dbg*/) override {
    return IOStatus::OK();
  }
  
  void SetPreallocationBlockSize(size_t /*size*/) override {}
  
  void GetPreallocationStatus(size_t* /*block_size*/, size_t* /*last_allocated_block*/) override {
    // 内存缓冲区不支持预分配
  }
  
  size_t GetUniqueId(char* /*id*/, size_t /*max_size*/) const override {
    return 0;  // 不支持
  }
  
  IOStatus InvalidateCache(size_t /*offset*/, size_t /*length*/) override {
    return IOStatus::OK();
  }
  
  bool IsSyncThreadSafe() const override { return true; }
  
  // use_fsync, SetWriteRateLimiter, SetPlaybackWriteRateLimiter 不在FSWritableFile接口中，已删除
  
  void SetWriteLifeTimeHint(Env::WriteLifeTimeHint /*hint*/) override {
    // 内存缓冲区阶段不需要设置hint，二次写入时会设置
  }
  
  Env::WriteLifeTimeHint GetWriteLifeTimeHint() override {
    return Env::WLTH_NOT_SET;
  }
  
  // GetFileName() is not in FSWritableFile interface, removed
};
}  // anonymous namespace

std::unique_ptr<FSWritableFile> TwoPhaseWriteManager::CreateMemoryWritableFile(uint64_t file_number) {
  std::string filename = "memory://" + std::to_string(file_number);
  return std::make_unique<MemoryWritableFile>(this, file_number, filename);
}

void TwoPhaseWriteManager::Shutdown() {
  if (!initialized_.load()) {
    return;
  }
  // 清理所有内存缓冲区
  {
    std::unique_lock<std::shared_mutex> lock(buffer_mutex_);
    memory_buffers_.clear();
  }
  initialized_.store(false);
}

}  // namespace ROCKSDB_NAMESPACE

