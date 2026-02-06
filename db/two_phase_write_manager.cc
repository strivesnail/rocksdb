//  Copyright (c) 2011-present, Facebook, Inc.  All rights reserved.
//  This source code is licensed under both the GPLv2 (found in the
//  COPYING file in the root directory) and Apache 2.0 License
//  (found in the LICENSE.Apache file in the root directory).

#include "db/two_phase_write_manager.h"

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
    fprintf(stderr, "[TwoPhaseWriteManager::Initialize] [VERIFY] Already initialized, returning OK\n");
    fflush(stderr);
    return Status::OK();
  }

  model_dir_ = model_dir;
  db_path_ = db_path;
  // 临时文件目录：使用 db_path/tmp
  tmp_dir_ = db_path_ + "/tmp";
  enable_phase2_ = enable_phase2;

  fprintf(stderr, "[TwoPhaseWriteManager::Initialize] [VERIFY] Creating tmp directory: %s\n", tmp_dir_.c_str());
  fflush(stderr);
  
  // 创建tmp目录 - NO ERROR TOLERANCE
  Env* env = Env::Default();
  Status s = env->CreateDirIfMissing(tmp_dir_);
  if (!s.ok()) {
    fprintf(stderr, "[TwoPhaseWriteManager::Initialize] [FATAL] Failed to create tmp directory %s: %s\n", 
            tmp_dir_.c_str(), s.ToString().c_str());
    fflush(stderr);
    return s;  // FAIL IMMEDIATELY
  }
  
  fprintf(stderr, "[TwoPhaseWriteManager::Initialize] [VERIFY] ✓ Tmp directory created\n");
  fflush(stderr);

  // 只有Phase 2才需要加载模型和启动后台线程
  if (enable_phase2_) {
    fprintf(stderr, "[TwoPhaseWriteManager::Initialize] [VERIFY] Phase2 enabled, loading models from: %s\n", model_dir.c_str());
    fflush(stderr);
    
    // 加载模型 - NO ERROR TOLERANCE
    s = LoadModels(model_dir);
    if (!s.ok()) {
      fprintf(stderr, "[TwoPhaseWriteManager::Initialize] [FATAL] Failed to load models from %s: %s\n",
              model_dir.c_str(), s.ToString().c_str());
      fflush(stderr);
      return s;  // FAIL IMMEDIATELY
    }
    
    fprintf(stderr, "[TwoPhaseWriteManager::Initialize] [VERIFY] ✓ Models loaded successfully\n");
    fflush(stderr);
  } else {
    fprintf(stderr, "[TwoPhaseWriteManager::Initialize] [VERIFY] Phase2 disabled, skipping model loading\n");
    fflush(stderr);
  }

  initialized_.store(true);
  fprintf(stderr, "[TwoPhaseWriteManager::Initialize] [VERIFY] ✓ Initialization completed successfully\n");
  fflush(stderr);
  return Status::OK();
}

bool TwoPhaseWriteManager::HandleFileCreation(uint64_t file_number, int level,
                                              const std::string& file_path) {
  fprintf(stderr, "[VERIFY] HandleFileCreation: file #%" PRIu64 " level=%d path=%s\n",
          file_number, level, file_path.c_str());
  fprintf(stderr, "[VERIFY] HandleFileCreation: enable_phase2=%d initialized=%d\n",
          enable_phase2_ ? 1 : 0, initialized_.load() ? 1 : 0);
  fflush(stderr);
  
  // Phase 1不需要两阶段写入，直接写入目标目录
  if (!enable_phase2_) {
    fprintf(stderr, "[VERIFY] HandleFileCreation: file #%" PRIu64 " - Phase2未启用，返回false（直接写入目标目录）\n", file_number);
    fflush(stderr);
    return false;  // 返回false表示不写tmp目录，直接写入目标目录
  }

  // NO ERROR TOLERANCE - 如果未初始化，这是严重错误
  if (!initialized_.load()) {
    fprintf(stderr, "[FATAL] HandleFileCreation: file #%" PRIu64 " - 未初始化！这是严重错误！\n", file_number);
    fflush(stderr);
    abort();  // FAIL IMMEDIATELY - NO FALLBACK
  }

  // Level 0文件直接写入目标目录（使用handle6），不写tmp
  if (level == 0) {
    fprintf(stderr, "[VERIFY] HandleFileCreation: file #%" PRIu64 " - Level 0，返回false（直接写入目标目录）\n", file_number);
    fflush(stderr);
    return false;  // 返回false表示不写tmp目录
  }

  // Phase 2的其他文件先写tmp目录
  fprintf(stderr, "[VERIFY] HandleFileCreation: file #%" PRIu64 " - Level %d，返回true（写入tmp目录）\n", file_number, level);
  fflush(stderr);
  return true;  // 返回true表示写tmp目录
}

Status TwoPhaseWriteManager::CollectFeaturesAndPredict(
    uint64_t file_number, int level, const std::vector<double>& features) {
  fprintf(stderr, "[VERIFY] CollectFeaturesAndPredict: file #%" PRIu64 " (level=%d, features_size=%zu, enable_phase2=%d, initialized=%d)\n",
          file_number, level, features.size(), enable_phase2_ ? 1 : 0, initialized_.load() ? 1 : 0);
  fflush(stderr);  // 强制刷新，确保日志立即输出
  
  // Phase 1不需要预测
  if (!enable_phase2_) {
    fprintf(stderr, "[VERIFY] CollectFeaturesAndPredict: file #%" PRIu64 " - Phase2未启用，返回OK（不进行预测）\n", file_number);
    return Status::OK();  // Phase 1直接返回OK，不进行预测
  }

  if (!initialized_.load() || level <= 0 || level > 6) {
    fprintf(stderr, "[VERIFY] CollectFeaturesAndPredict: file #%" PRIu64 " - ✗ 无效参数 (initialized=%d, level=%d)\n",
            file_number, initialized_.load() ? 1 : 0, level);
    return Status::InvalidArgument("Invalid level or not initialized");
  }

  if (features.size() != 69) {
    fprintf(stderr, "[VERIFY] CollectFeaturesAndPredict: file #%" PRIu64 " - ✗ 特征数量错误 (%zu != 69)\n",
            file_number, features.size());
    return Status::InvalidArgument("Invalid feature count");
  }

  // 性能测量：ML预测
  fprintf(stderr, "[VERIFY] CollectFeaturesAndPredict: file #%" PRIu64 " - 调用PredictLifetime (level=%d, features_size=%zu)\n",
          file_number, level, features.size());
  
  uint64_t t0_predict = Env::Default()->NowMicros();
  double predicted_lifetime = PredictLifetime(level, features);
  uint64_t t1_predict = Env::Default()->NowMicros();
  fprintf(stderr, "[PERF] CollectFeaturesAndPredict: file #%" PRIu64 " - PredictLifetime耗时: %.3f ms\n",
          file_number, (t1_predict - t0_predict) / 1000.0);
  
  fprintf(stderr, "[VERIFY] CollectFeaturesAndPredict: file #%" PRIu64 " - PredictLifetime返回: %.6f\n",
          file_number, predicted_lifetime);
  
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

  // 映射到handle
  int target_handle = MapLifetimeToHandle(predicted_lifetime, level);

  // 构建文件路径（使用RocksDB的文件名格式，带前导零）
  // 格式：000012.sst（6位数字，不足补0）
  char tmp_filename[32];
  char target_filename[32];
  snprintf(tmp_filename, sizeof(tmp_filename), "%06" PRIu64 ".sst", file_number);
  snprintf(target_filename, sizeof(target_filename), "%06" PRIu64 ".sst", file_number);
  std::string tmp_file_path = tmp_dir_ + "/" + tmp_filename;
  std::string target_file_path = db_path_ + "/" + target_filename;

  // 保存元数据（写操作，使用unique_lock）
  {
    std::unique_lock<std::shared_mutex> lock(metadata_mutex_);
    file_metadata_[file_number] = {
        file_number,
        level,
        current_time,
        predicted_lifetime,
        tmp_file_path,
        target_file_path,
        target_handle
    };
  }

  fprintf(stderr, "[VERIFY] CollectFeaturesAndPredict: file #%" PRIu64 " - ✓ 预测成功 (lifetime=%.3f秒, handle=%d)\n",
          file_number, predicted_lifetime, target_handle);

  return Status::OK();
}

Status TwoPhaseWriteManager::RewriteFileToTargetHandle(uint64_t file_number) {
  fprintf(stderr, "[VERIFY] RewriteFileToTargetHandle: file #%" PRIu64 "\n", file_number);
  fprintf(stderr, "[VERIFY] RewriteFileToTargetHandle: enable_phase2=%d initialized=%d\n",
          enable_phase2_ ? 1 : 0, initialized_.load() ? 1 : 0);
  fflush(stderr);
  
  // Phase 1不需要重新写入
  if (!enable_phase2_) {
    fprintf(stderr, "[VERIFY] RewriteFileToTargetHandle: file #%" PRIu64 " - Phase2未启用，返回OK（不进行重新写入）\n", file_number);
    fflush(stderr);
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

  fprintf(stderr, "[VERIFY] RewriteFileToTargetHandle: file #%" PRIu64 " - metadata found (level=%d, handle=%d, tmp_path=%s, target_path=%s)\n",
          file_number, metadata.level, metadata.target_handle, metadata.tmp_file_path.c_str(), metadata.target_file_path.c_str());
  fflush(stderr);

  // Level 0文件已经在目标目录，不需要重新写入
  if (metadata.level == 0) {
    fprintf(stderr, "[VERIFY] RewriteFileToTargetHandle: file #%" PRIu64 " - Level 0，跳过重新写入\n", file_number);
    fflush(stderr);
    return Status::OK();
  }

  // 直接从tmp_dir_和file_number构建临时文件路径
  // 第一次写入后，文件肯定在db1/tmp/中，文件名就是file_number.sst（6位零填充）
  char tmp_filename[32];
  snprintf(tmp_filename, sizeof(tmp_filename), "%06" PRIu64 ".sst", file_number);
  std::string src_path = tmp_dir_ + "/" + tmp_filename;
  
  // 构建正确的目标路径（使用RocksDB的文件名格式，带前导零）
  // 格式：000012.sst（6位数字，不足补0）
  char target_filename[32];
  snprintf(target_filename, sizeof(target_filename), "%06" PRIu64 ".sst", file_number);
  std::string correct_target_path = db_path_ + "/" + target_filename;
  
  // 减少日志输出以提高性能（只在DEBUG模式下输出）
  // fprintf(stderr, "[TwoPhaseWrite] RewriteFileToTargetHandle: file #%" PRIu64 " - 目标路径: %s (metadata中的路径: %s)\n",
  //         file_number, correct_target_path.c_str(), metadata.target_file_path.c_str());
  // fflush(stderr);

  fprintf(stderr, "[VERIFY] RewriteFileToTargetHandle: file #%" PRIu64 " - 开始移动文件 (src=%s, dst=%s, handle=%d)\n",
          file_number, src_path.c_str(), correct_target_path.c_str(), metadata.target_handle);
  fflush(stderr);
  
  // 原子性移动文件（使用正确的目标路径） - NO ERROR TOLERANCE
  Status s = MoveFileAtomically(src_path, 
                                correct_target_path, 
                                metadata.target_handle);
  if (!s.ok()) {
    fprintf(stderr, "[FATAL] RewriteFileToTargetHandle: file #%" PRIu64 " - 移动文件失败: %s\n", 
            file_number, s.ToString().c_str());
    fprintf(stderr, "[FATAL] RewriteFileToTargetHandle: src_path=%s\n", src_path.c_str());
    fprintf(stderr, "[FATAL] RewriteFileToTargetHandle: target_path=%s\n", correct_target_path.c_str());
    fprintf(stderr, "[FATAL] RewriteFileToTargetHandle: handle=%d\n", metadata.target_handle);
    fflush(stderr);
    abort();  // FAIL IMMEDIATELY - NO FALLBACK
  }

  fprintf(stderr, "[VERIFY] RewriteFileToTargetHandle: file #%" PRIu64 " - ✓ 文件移动成功\n", file_number);
  fflush(stderr);

  return Status::OK();
}

int TwoPhaseWriteManager::MapLifetimeToHandle(double predicted_lifetime_seconds,
                                              int level) {
  // Level 0文件直接返回handle6
  if (level == 0) {
    return 6;
  }

  // 根据预测的生命周期映射（与画图时使用的映射一致）
  // Handle 6: [0, 8) seconds
  // Handle 7: [8, 12) seconds
  // Handle 8: [12, 30) seconds
  // Handle 9: [30, 100) seconds
  // Handle 10: [100, 300) seconds
  // Handle 11: [300, 600) seconds
  // Handle 12: [600, +∞) seconds
  if (predicted_lifetime_seconds < 8.0) {
    return 6;
  } else if (predicted_lifetime_seconds < 12.0) {
    return 7;
  } else if (predicted_lifetime_seconds < 30.0) {
    return 8;
  } else if (predicted_lifetime_seconds < 100.0) {
    return 9;
  } else if (predicted_lifetime_seconds < 300.0) {
    return 10;
  } else if (predicted_lifetime_seconds < 600.0) {
    return 11;
  } else {
    return 12;
  }
}

Status TwoPhaseWriteManager::MoveFileAtomically(const std::string& src_path,
                                                const std::string& dst_path,
                                                int target_handle) {
  // 使用和第一次写入完全一样的方式：使用 RocksDB 的 WritableFileWriter
  // 只是数据源不同：从临时文件读取，而不是从 compaction 结果读取
  
  Env* env = Env::Default();
  FileSystem* fs = env->GetFileSystem().get();
  
  // 获取源文件大小（用于性能分析）
  uint64_t src_file_size = 0;
  Status size_status = env->GetFileSize(src_path, &src_file_size);
  if (!size_status.ok()) {
    // 如果获取大小失败，继续执行，但记录警告
    fprintf(stderr, "[PERF] MoveFileAtomically - 警告：无法获取源文件大小: %s\n",
            size_status.ToString().c_str());
  }
  
  // 性能测量：打开源文件
  uint64_t t0_open_src = Env::Default()->NowMicros();
  FileOptions src_options;
  std::unique_ptr<FSSequentialFile> src_file;
  IOStatus io_s = fs->NewSequentialFile(src_path, src_options, &src_file, nullptr);
  if (!io_s.ok()) {
    return io_s;
  }
  uint64_t t1_open_src = Env::Default()->NowMicros();
  fprintf(stderr, "[PERF] MoveFileAtomically - 文件大小: %.2f MB, 打开源文件耗时: %.3f ms\n",
          src_file_size / 1024.0 / 1024.0, (t1_open_src - t0_open_src) / 1000.0);
  
  // 性能测量：创建目标文件
  uint64_t t0_create_dst = Env::Default()->NowMicros();
  // 创建目标文件（使用和第一次写入相同的 FileOptions）
  // 注意：目标文件不应该已存在（file_number是唯一的），不需要删除
  FileOptions dst_options;
  dst_options.write_hint = static_cast<Env::WriteLifeTimeHint>(target_handle);
  
  std::unique_ptr<FSWritableFile> dst_file;
  io_s = fs->NewWritableFile(dst_path, dst_options, &dst_file, nullptr);
  if (!io_s.ok()) {
    return io_s;
  }
  
  // 设置 write hint（FDP handle）
  dst_file->SetWriteLifeTimeHint(dst_options.write_hint);
  uint64_t t1_create_dst = Env::Default()->NowMicros();
  fprintf(stderr, "[PERF] MoveFileAtomically - 创建目标文件耗时: %.3f ms\n",
          (t1_create_dst - t0_create_dst) / 1000.0);
  
  // 创建 WritableFileWriter（和第一次写入完全一样）
  std::unique_ptr<WritableFileWriter> writer(
      new WritableFileWriter(std::move(dst_file), dst_path, dst_options,
                            nullptr, nullptr, nullptr, Histograms::SST_WRITE_MICROS,
                            std::vector<std::shared_ptr<EventListener>>(),
                            nullptr, false, false));
  
  // 创建 SequentialFileReader（从临时文件读取）
  std::unique_ptr<SequentialFileReader> reader(
      new SequentialFileReader(std::move(src_file), src_path, nullptr));
  
  // 性能测量：流式传输
  uint64_t t0_copy = Env::Default()->NowMicros();
  // 流式传输：从临时文件读取，写入目标文件（边读边写，和 RewriteTempOutputFile 完全一样）
  // 使用4MB缓冲区，已经是边读边写的最优方式
  constexpr size_t kReadBufferSize = 4 * 1024 * 1024;  // 4MB
  std::unique_ptr<char[]> buffer(new char[kReadBufferSize]);
  Slice slice;
  IOOptions io_opts;
  
  uint64_t total_bytes = 0;
  while (true) {
    io_s = reader->Read(kReadBufferSize, &slice, buffer.get(), Env::IO_HIGH);
    if (!io_s.ok()) {
      return io_s;
    }
    if (slice.empty()) {
      break;
    }
    total_bytes += slice.size();
    io_s = writer->Append(io_opts, slice);
    if (!io_s.ok()) {
      return io_s;
    }
  }
  uint64_t t1_copy = Env::Default()->NowMicros();
  double copy_time_ms = (t1_copy - t0_copy) / 1000.0;
  double copy_speed_mbps = (total_bytes / 1024.0 / 1024.0) / (copy_time_ms / 1000.0);
  fprintf(stderr, "[PERF] MoveFileAtomically - 流式传输: %.3f ms, 大小: %.2f MB, 速度: %.2f MB/s (文件大小: %.2f MB)\n",
          copy_time_ms, total_bytes / 1024.0 / 1024.0, copy_speed_mbps, src_file_size / 1024.0 / 1024.0);
  
  // 性能测量：关闭文件
  uint64_t t0_close = Env::Default()->NowMicros();
  // 关闭文件（确保文件完全写入并可见）
  // 注意：不调用Sync，Close会确保数据写入到文件系统
  io_s = writer->Close(io_opts);
  if (!io_s.ok()) {
    return io_s;
  }
  uint64_t t1_close = Env::Default()->NowMicros();
  double close_time_ms = (t1_close - t0_close) / 1000.0;
  fprintf(stderr, "[PERF] MoveFileAtomically - 关闭文件耗时: %.3f ms (文件大小: %.2f MB, 关闭速度: %.2f MB/s)\n",
          close_time_ms, src_file_size / 1024.0 / 1024.0,
          (src_file_size / 1024.0 / 1024.0) / (close_time_ms / 1000.0));
  
  // 性能测量：删除临时文件
  uint64_t t0_delete = Env::Default()->NowMicros();
  // 删除原文件（tmp目录中的）
  // 注意：不检查FileExists，Close后文件应该已经可见
  Status delete_s = env->DeleteFile(src_path);
  if (!delete_s.ok()) {
    // 不返回错误，因为文件已经成功移动到目标位置
  }
  uint64_t t1_delete = Env::Default()->NowMicros();
  fprintf(stderr, "[PERF] MoveFileAtomically - 删除临时文件耗时: %.3f ms\n",
          (t1_delete - t0_delete) / 1000.0);
  
  uint64_t t_total = t1_delete - t0_open_src;
  fprintf(stderr, "[PERF] MoveFileAtomically - 总耗时: %.3f ms (文件大小: %.2f MB, 平均速度: %.2f MB/s)\n",
          t_total / 1000.0, src_file_size / 1024.0 / 1024.0,
          (src_file_size / 1024.0 / 1024.0) / (t_total / 1000.0 / 1000.0));

  return Status::OK();
}

// 已移除：所有删除队列和移动队列相关函数 - 不再管理文件删除

Status TwoPhaseWriteManager::LoadModels(const std::string& model_dir) {
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
  
  // 使用现有的预测函数
  double result = PredictFileLifetimePythonByLevel(features.data(), features.size(), level);
  
  fprintf(stderr, "[VERIFY] PredictLifetime: PredictFileLifetimePythonByLevel返回: %.6f\n",
          result);
  
  if (result < 0) {
    fprintf(stderr, "[ERROR] PredictLifetime: 预测函数返回负数！这表示预测失败。\n");
    fprintf(stderr, "  - level: %d\n", level);
    fprintf(stderr, "  - features.size(): %zu\n", features.size());
    fprintf(stderr, "  - 返回值: %.6f\n", result);
    
    // 检查特征值是否有效
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
  }
  
  return result;
#else
  fprintf(stderr, "[ERROR] PredictLifetime: ROCKSDB_ML_PREDICT_PYTHON未定义，不支持预测\n");
  return -1.0;  // 不支持
#endif
}

void TwoPhaseWriteManager::Shutdown() {
  if (!initialized_.load()) {
    return;
  }
  // 已移除：不再需要停止删除队列和移动队列线程
  initialized_.store(false);
}

}  // namespace ROCKSDB_NAMESPACE

