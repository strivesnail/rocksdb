//  Copyright (c) 2011-present, Facebook, Inc.  All rights reserved.
//  This source code is licensed under both the GPLv2 (found in the
//  COPYING file in the root directory) and Apache 2.0 License
//  (found in the LICENSE.Apache file in the root directory).

#include "db/two_phase_write_manager.h"

#include "rocksdb/io_status.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cinttypes>
#include <cstdio>
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
#include <unordered_map>  // for thread_local std::unordered_map

#include "rocksdb/env.h"
#include "rocksdb/status.h"
#include "rocksdb/slice.h"
#include "rocksdb/file_system.h"
#include "util/string_util.h"
#include "db/event_helpers_ml_features.h"
#include "db/compaction/subcompaction_state.h"
#include "db/compaction/compaction.h"
#include "db/column_family.h"
#include "db/version_set.h"
#include "monitoring/instrumented_mutex.h"
#ifdef ROCKSDB_ML_PREDICT_ONNX
#include "db/onnx_predictor.h"
#endif
#ifdef ROCKSDB_ML_PREDICT_PYTHON
#include "tools/ml_predict_python.h"
#endif
#include "file/writable_file_writer.h"
#include "file/sequence_file_reader.h"
#include "file/file_util.h"
#include "file/filename.h"
#include "monitoring/histogram.h"
#include "logging/logging.h"

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
      info_log_(nullptr),
      initialized_(false) {
  memset(models_, 0, sizeof(models_));
  memset(scalers_, 0, sizeof(scalers_));
}

TwoPhaseWriteManager::~TwoPhaseWriteManager() {
  Shutdown();
}

Status TwoPhaseWriteManager::Initialize(const std::string& model_dir,
                                        const std::string& db_path,
                                        bool enable_phase2,
                                        Logger* info_log) {
  
  if (initialized_.load()) {
    return Status::OK();
  }

  model_dir_ = model_dir;
  db_path_ = db_path;
  enable_phase2_ = enable_phase2;
  info_log_ = info_log;
  use_hash_handle_ = false;

  if (enable_phase2_) {
    const char* hash_env = std::getenv("ROCKSDB_HASH_HANDLE");
    if (hash_env == nullptr || hash_env[0] == '\0' || hash_env[1] != '\0' ||
        (hash_env[0] != '0' && hash_env[0] != '1')) {
      return Status::InvalidArgument(
          "ROCKSDB_ENABLE_PHASE2=1 requires ROCKSDB_HASH_HANDLE to be exactly "
          "\"0\" (level-base) or \"1\" (hash-base). "
          "Unset, empty, or any other value is an error (no fallback).");
    }
    use_hash_handle_ = (hash_env[0] == '1');
  }

  // Phase2需要加载模型进行预测
  if (enable_phase2_) {
    Status s = LoadModels(model_dir);
    if (!s.ok()) {
      ROCKS_LOG_ERROR(info_log_, "[TwoPhaseWriteManager] Initialize: 模型加载失败: %s", s.ToString().c_str());
      return s;
    }
    LoadFeatureSubset(model_dir);
    if (info_log_) {
      ROCKS_LOG_INFO(info_log_,
                     "[TwoPhaseWriteManager] initialized enable_phase2=1 "
                     "ROCKSDB_HASH_HANDLE=%s",
                     use_hash_handle_ ? "1 (hash-base)" : "0 (level-base)");
      const char* reg = std::getenv("ROCKSDB_PHASE2_REGISTER_METADATA");
      if (reg != nullptr && reg[0] == '0' && reg[1] == '\0') {
        ROCKS_LOG_WARN(info_log_,
                       "[TwoPhaseWriteManager][FALLBACK/config] "
                       "ROCKSDB_PHASE2_REGISTER_METADATA=0: compaction output "
                       "metadata registration is OFF.");
      }
      const char* rew = std::getenv("ROCKSDB_PHASE2_REWRITE");
      if (rew != nullptr && rew[0] == '0' && rew[1] == '\0') {
        ROCKS_LOG_WARN(info_log_,
                       "[TwoPhaseWriteManager][FALLBACK/config] "
                       "ROCKSDB_PHASE2_REWRITE=0: Phase2 rewrite to target "
                       "handle is OFF.");
      }
    }
  }

  initialized_.store(true);
  return Status::OK();
}

bool TwoPhaseWriteManager::HandleFileCreation(uint64_t file_number, int level,
                                              const std::string& /*file_path*/) {
  // Phase1：所有文件都直接写入目标目录
  if (!enable_phase2_) {
    return false;
  }
#if defined(ROCKSDB_ML_PREDICT_PYTHON) || defined(ROCKSDB_ML_PREDICT_ONNX)
  // Phase2 + ML：Level 0 固定 handle 6，Level 6 固定 handle 12；Level 1–5 按预测寿命映射到 handle 7–11
  // Level 0: 直接写入 handle 6（无预测）
  // Level 6: 直接写入 handle 12（无预测）
  // Level 1–5: 写入内存缓冲区，特征计算 + ML 预测后按寿命映射到 handle 7–11
  if (level == 0) {
    return false;
  }
  if (level == 6) {
    return false;
  }
  if (level < 1 || level > 5) {
    return false;
  }

  // Level 1–5 写入内存缓冲区
  Status s = CreateMemoryBuffer(file_number);
  if (!s.ok()) {
    ROCKS_LOG_ERROR(info_log_, "[TwoPhaseWriteManager] HandleFileCreation: file #%" PRIu64 " (level=%d) - 创建内存缓冲区失败: %s",
            file_number, level, s.ToString().c_str());
    return false;  // 创建失败，回退到直接写入
  }
  
  return true;  // 写入内存缓冲区
#else
  // Phase2 但未定义 ML 预测后端：仅自定义 compaction，不二次写入，全部直接写
  (void)file_number;
  (void)level;
  return false;
#endif
}

Status TwoPhaseWriteManager::CollectFeaturesAndPredict(
    uint64_t file_number, int level, const std::vector<double>& features) {
  // Phase 1不需要预测
  if (!enable_phase2_) {
    return Status::OK();  // Phase 1直接返回OK，不进行预测
  }

  // 验证：检查level是否有效
  if (level < 0 || level > 6) {
    ROCKS_LOG_ERROR(info_log_, "[TwoPhaseWriteManager] CollectFeaturesAndPredict: file #%" PRIu64 " - 无效的level: %d（有效范围: 0-6）",
            file_number, level);
    return Status::InvalidArgument("Invalid level: " + std::to_string(level));
  }

  double predicted_lifetime;
  int target_handle;
#if defined(ROCKSDB_ML_PREDICT_PYTHON) || defined(ROCKSDB_ML_PREDICT_ONNX)
  // 计时：ML 预测
  uint64_t t_predict_start = Env::Default()->NowMicros();
  
  // 调用预测函数获取预测的生命周期
  predicted_lifetime = PredictLifetime(level, features);
  if (predicted_lifetime < 0) {
    ROCKS_LOG_ERROR(info_log_, "[TwoPhaseWriteManager] CollectFeaturesAndPredict: file #%" PRIu64 " - 预测失败，predicted_lifetime=%.2f",
            file_number, predicted_lifetime);
    return Status::Corruption("Prediction failed for file " + std::to_string(file_number));
  }
  
  uint64_t t_predict_end = Env::Default()->NowMicros();
  
  target_handle = MapLifetimeToHandle(predicted_lifetime, level);
  (void)t_predict_start;
  (void)t_predict_end;
#else
  // 无 ML 预测：直接按 level 映射 handle（仅用于自定义 compaction 的元数据，不二次写入）
  (void)features;
  predicted_lifetime = 0.0;
  target_handle = level + 6;
#endif
  
  // 验证：检查handle是否在有效范围内 [6, 12]
  if (target_handle < 6 || target_handle > 12) {
    ROCKS_LOG_ERROR(info_log_, "[TwoPhaseWriteManager] CollectFeaturesAndPredict: file #%" PRIu64 " - handle=%d 超出有效范围 [6, 12]！level=%d, predicted_lifetime=%.2f",
            file_number, target_handle, level, predicted_lifetime);
    return Status::Corruption("Invalid handle: " + std::to_string(target_handle) + 
                             " for level: " + std::to_string(level));
  }

  uint64_t current_time = Env::Default()->NowMicros();

  // 保存元数据（写操作，使用unique_lock）
  {
    std::unique_lock<std::shared_mutex> lock(metadata_mutex_);
    file_metadata_[file_number] = { current_time, target_handle, level };
  }

  return Status::OK();
}

Status TwoPhaseWriteManager::RewriteFileToTargetHandle(uint64_t file_number) {
  // Phase2进行二次写入：从内存缓冲区读取数据，写入目标文件（带handle）
  if (!enable_phase2_) {
    return Status::OK();  // Phase1不需要二次写入
  }

  int target_handle;
  {
    std::shared_lock<std::shared_mutex> lock(metadata_mutex_);
    auto it = file_metadata_.find(file_number);
    if (it == file_metadata_.end()) {
      ROCKS_LOG_ERROR(info_log_, "[TwoPhaseWriteManager] RewriteFileToTargetHandle: file #%" PRIu64 " - 元数据不存在",
              file_number);
      return Status::NotFound("File metadata not found for file " + std::to_string(file_number));
    }
    target_handle = it->second.target_handle;
  }

  std::string buffer_data;
  Status s = GetMemoryBufferData(file_number, &buffer_data);
  if (!s.ok()) {
    ROCKS_LOG_ERROR(info_log_, "[TwoPhaseWriteManager] RewriteFileToTargetHandle: file #%" PRIu64 " - 无法获取内存缓冲区数据: %s",
            file_number, s.ToString().c_str());
    return s;
  }

  char target_filename[32];
  snprintf(target_filename, sizeof(target_filename), "%06" PRIu64 ".sst", file_number);
  std::string target_path = db_path_ + "/" + target_filename;
  s = WriteMemoryBufferToFile(file_number, buffer_data, target_path, target_handle);
  if (!s.ok()) {
    ROCKS_LOG_ERROR(info_log_, "[TwoPhaseWriteManager] RewriteFileToTargetHandle: file #%" PRIu64 " - 写入目标文件失败: %s",
            file_number, s.ToString().c_str());
    return s;
  }

  ReleaseMemoryBuffer(file_number);
  return Status::OK();
}

// Handle 阈值定义（秒）- 两套：写入时 vs Compaction 时
// 写入时（MapLifetimeToHandle）：L0→handle6, L6→handle12；L1–5 按预测寿命映射到 handle 7–11：
//   区间 [0,5), [5,10), [10,15), [15,60), [60,+∞) 分别对应 handle 7,8,9,10,11
// Compaction 时：仅上限，用于 GetHandleThreshold（sort_value = elapsed_time - threshold）
namespace {
  // Level 1–5 专用：MapLifetimeToHandle 的写入时上界（实际使用）
  constexpr double kWriteL15_Handle7Upper = 5.0;
  constexpr double kWriteL15_Handle8Upper = 10.0;
  constexpr double kWriteL15_Handle9Upper = 15.0;
  constexpr double kWriteL15_Handle10Upper = 60.0;

  // Compaction 时仅上限（GetHandleThreshold / GetHandleBounds 推导用）
  // handle 6 无阈值（仅 L0 使用，L0 不参与自定义 compaction 排序）
  constexpr double kCompactionHandle6Threshold = std::numeric_limits<double>::max();
  constexpr double kCompactionHandle7Threshold = 5.0;    // Level 1
  constexpr double kCompactionHandle8Threshold = 10.0;   // Level 2
  constexpr double kCompactionHandle9Threshold = 15.0;   // Level 3
  constexpr double kCompactionHandle10Threshold = 60.0;  // Level 4
  constexpr double kCompactionHandle11Threshold = 100.0; // Level 5
  constexpr double kCompactionHandle12Threshold = std::numeric_limits<double>::max();

  // 用 file_number + level 做 hash，将 level 1–6 均匀映射到 handle 7–12（7 + h%6）
  int HashToHandle(uint64_t file_number, int level) {
    if (level >= 1 && level <= 6) {
      uint64_t h = file_number * 31ULL + static_cast<uint64_t>(level);
      return 7 + static_cast<int>(h % 6);
    }
    return 7;  // 不应走到（L0 由调用方固定 6）
  }
}

int TwoPhaseWriteManager::GetHandleByHash(uint64_t file_number, int level) const {
  return HashToHandle(file_number, level);
}

int TwoPhaseWriteManager::GetTargetHandleForCompactionOutputMetadata(
    uint64_t file_number, int level) const {
  if (level == 0) {
    return 6;
  }
  if (level < 1 || level > 6) {
    if (info_log_) {
      static std::atomic<unsigned> bad_level_logs{0};
      unsigned c = bad_level_logs.fetch_add(1, std::memory_order_relaxed);
      if (c < 16u) {
        ROCKS_LOG_WARN(
            info_log_,
            "[TwoPhaseWriteManager][FALLBACK] "
            "GetTargetHandleForCompactionOutputMetadata: level=%d outside [0,6], "
            "using LevelToHandle (file #%" PRIu64 ").",
            level, file_number);
      }
    }
    return LevelToHandle(level);
  }
  if (!enable_phase2_) {
    return LevelToHandle(level);
  }
  if (use_hash_handle_) {
    return GetHandleByHash(file_number, level);
  }
  return LevelToHandle(level);
}

int TwoPhaseWriteManager::MapLifetimeToHandle(double predicted_lifetime_seconds,
                                              int level) {
  // Level 0 永远写进 handle 6
  if (level == 0) {
    return 6;
  }
  // Level 6（最底层）永远写进 handle 12
  if (level == 6) {
    return 12;
  }

  // Level 1–5：仅映射到 handle 7–11（L0 用 handle 6，不写 handle 12）
  if (predicted_lifetime_seconds < kWriteL15_Handle7Upper) {
    return 7;
  }
  if (predicted_lifetime_seconds < kWriteL15_Handle8Upper) {
    return 8;
  }
  if (predicted_lifetime_seconds < kWriteL15_Handle9Upper) {
    return 9;
  }
  if (predicted_lifetime_seconds < kWriteL15_Handle10Upper) {
    return 10;
  }
  return 11;
}

int TwoPhaseWriteManager::LevelToHandle(int level) {
  if (level == 0) {
    return 6;  // L0 写进 handle 6
  }
  if (level >= 1 && level <= 6) {
    return 6 + level;
  }
  return -1;
}

int TwoPhaseWriteManager::GetFileTargetHandle(uint64_t file_number) const {
  // 从file_metadata_中查找文件的target_handle
  std::shared_lock<std::shared_mutex> lock(metadata_mutex_);
  auto it = file_metadata_.find(file_number);
  if (it != file_metadata_.end()) {
    return it->second.target_handle;
  }
  // 如果找不到，返回-1表示未知
  return -1;
}

double TwoPhaseWriteManager::GetHandleThreshold(int handle) const {
  // 返回该 handle 的 compaction 上限（用于 sort_value = elapsed_time - threshold）
  switch (handle) {
    case 6: return kCompactionHandle6Threshold;
    case 7: return kCompactionHandle7Threshold;
    case 8: return kCompactionHandle8Threshold;
    case 9: return kCompactionHandle9Threshold;
    case 10: return kCompactionHandle10Threshold;
    case 11: return kCompactionHandle11Threshold;
    case 12: return kCompactionHandle12Threshold;
    default:
      ROCKS_LOG_WARN(info_log_, "[TwoPhaseWriteManager] GetHandleThreshold: invalid handle %d, using handle 6 threshold", handle);
      return kCompactionHandle6Threshold;
  }
}

double FileLifetimeInfo::GetCurrentAge() const {
  if (creation_time == 0) return 0.0;
  uint64_t now = Env::Default()->NowMicros();
  return (now - creation_time) / 1000000.0;
}

double FileLifetimeInfo::GetExcessOverThreshold() const {
  double age = GetCurrentAge();
  return age - threshold_seconds;
}

void TwoPhaseWriteManager::RegisterFileMetadata(uint64_t file_number, int level,
                                               int target_handle,
                                               uint64_t file_size) {
  if (!initialized_.load() || !enable_phase2_ ||
      target_handle < 6 || target_handle > 12) {
    return;
  }
  uint64_t creation_time = Env::Default()->NowMicros();
  std::unique_lock<std::shared_mutex> lock(metadata_mutex_);
  file_metadata_[file_number] = { creation_time, target_handle, level };
  ROCKS_LOG_INFO(info_log_,
                 "[RegisterFileMetadata] file #%" PRIu64 " level=%d handle=%d registered (creation_time=%" PRIu64 ", size=%" PRIu64 ")",
                 file_number, level, target_handle, creation_time, file_size);
  // 统一格式：每个写入文件一条，便于统计与解析
  ROCKS_LOG_INFO(info_log_,
                 "[FILE_WRITE] file #%" PRIu64 " level=%d handle=%d creation_time=%" PRIu64 " size=%" PRIu64,
                 file_number, level, target_handle, creation_time, file_size);
}

bool TwoPhaseWriteManager::ShouldRegisterCompactionOutputMetadata() const {
  if (!initialized_.load() || !enable_phase2_) return false;
  const char* v = std::getenv("ROCKSDB_PHASE2_REGISTER_METADATA");
  if (v != nullptr && std::string(v) == "0") return false;
  return true;
}

void TwoPhaseWriteManager::RegisterCompactionOutputFileMetadata(
    uint64_t file_number, int level, uint64_t file_size) {
  int target_handle = GetTargetHandleForCompactionOutputMetadata(file_number, level);
  if (target_handle < 0) return;
  RegisterFileMetadata(file_number, level, target_handle, file_size);
}

bool TwoPhaseWriteManager::ShouldDoPhase2Rewrite() const {
  const char* v = std::getenv("ROCKSDB_PHASE2_REWRITE");
  if (v != nullptr && std::string(v) == "0") return false;
  return true;
}

void TwoPhaseWriteManager::MaybeRegisterCompactionOutputMetadata(
    uint64_t file_number, int level, uint64_t file_size) {
  if (!ShouldRegisterCompactionOutputMetadata()) return;
  RegisterCompactionOutputFileMetadata(file_number, level, file_size);
}

void TwoPhaseWriteManager::MaybeDoPhase2Rewrite(uint64_t file_number) {
  if (ShouldDoPhase2Rewrite()) {
    Status s = RewriteFileToTargetHandle(file_number);
    if (!s.ok() && info_log_) {
      ROCKS_LOG_WARN(info_log_,
                    "[FALLBACK] Phase2 rewrite failed for file #%" PRIu64
                    ": %s (buffer may be released without full rewrite).",
                    file_number, s.ToString().c_str());
    }
    if (!s.ok()) {
      ReleaseMemoryBuffer(file_number);
    }
  } else {
    ReleaseMemoryBuffer(file_number);
  }
}

uint64_t TwoPhaseWriteManager::IncrementAndGetTrivialMoveCount(uint64_t file_number) {
  std::unique_lock<std::shared_mutex> lock(metadata_mutex_);
  uint64_t& count = trivial_move_count_[file_number];
  count++;
  return count;
}

bool TwoPhaseWriteManager::IsFileTooFar(uint64_t file_number) const {
  std::shared_lock<std::shared_mutex> lock(metadata_mutex_);
  auto it = file_metadata_.find(file_number);
  if (it == file_metadata_.end()) return false;
  uint64_t creation_time = it->second.creation_time;
  if (creation_time == 0) return false;
  int handle = it->second.target_handle;
  if (handle < 6 || handle > 12) return false;
  double threshold = GetHandleThreshold(handle);
  uint64_t now = Env::Default()->NowMicros();
  double age_sec = (now - creation_time) / 1000000.0;
  return age_sec > threshold;
}

std::vector<FileLifetimeInfo> TwoPhaseWriteManager::GetAllTooFarFiles(
    int level, VersionStorageInfo* vstorage) {
  std::vector<FileLifetimeInfo> result;
  if (vstorage == nullptr || level < 0 || level >= vstorage->num_levels()) {
    return result;
  }
  const std::vector<FileMetaData*>& files = vstorage->LevelFiles(level);
  uint64_t now = Env::Default()->NowMicros();
  std::shared_lock<std::shared_mutex> lock(metadata_mutex_);
  int with_metadata = 0;
  int skipped_being_compacted = 0;
  for (FileMetaData* f : files) {
    if (f == nullptr) continue;
    if (f->being_compacted) { skipped_being_compacted++; continue; }
    uint64_t file_number = f->fd.GetNumber();
    auto it = file_metadata_.find(file_number);
    if (it == file_metadata_.end()) continue;
    with_metadata++;
    uint64_t creation_time = it->second.creation_time;
    if (creation_time == 0) continue;
    int handle = it->second.target_handle;
    // 仅 L1–L6（hash 模式）会进入本路径，只应出现 handle 7–12
    if (handle < 7 || handle > 12) {
      if (handle >= 6 && handle <= 12) {
        ROCKS_LOG_WARN(info_log_,
            "[GetAllTooFarFiles] file #%" PRIu64 " level=%d handle=%d unexpected (expected 7-12); possible mapping bug",
            file_number, level, handle);
      }
      continue;
    }
    double threshold = GetHandleThreshold(handle);
    double age_sec = (now - creation_time) / 1000000.0;
    if (age_sec <= threshold) continue;
    FileLifetimeInfo info;
    info.file_number = file_number;
    info.handle = handle;
    info.creation_time = creation_time;
    info.threshold_seconds = threshold;
    result.push_back(info);
  }
  ROCKS_LOG_INFO(info_log_,
      "[GetAllTooFarFiles] level=%d: files_on_level=%zu, with_metadata=%d, being_compacted_skipped=%d, too_far_count=%zu",
      level, files.size(), with_metadata, skipped_being_compacted, result.size());
  return result;
}

void TwoPhaseWriteManager::GetHandleBounds(int handle, double* lower_bound, double* upper_bound) const {
  // 由 compaction 上限推导区间（仅用于兼容调用方，init 已不打印区间）
  // A 方案：无 case 6，handle 6 由 default 处理（upper_bound = kCompactionHandle6Threshold = +inf）
  if (lower_bound == nullptr || upper_bound == nullptr) {
    return;
  }
  switch (handle) {
    case 7:
      *lower_bound = 0.0;
      *upper_bound = kCompactionHandle7Threshold;
      break;
    case 8:
      *lower_bound = kCompactionHandle7Threshold;
      *upper_bound = kCompactionHandle8Threshold;
      break;
    case 9:
      *lower_bound = kCompactionHandle8Threshold;
      *upper_bound = kCompactionHandle9Threshold;
      break;
    case 10:
      *lower_bound = kCompactionHandle9Threshold;
      *upper_bound = kCompactionHandle10Threshold;
      break;
    case 11:
      *lower_bound = kCompactionHandle10Threshold;
      *upper_bound = kCompactionHandle11Threshold;
      break;
    case 12:
      *lower_bound = kCompactionHandle11Threshold;
      *upper_bound = kCompactionHandle12Threshold;
      break;
    default:
      ROCKS_LOG_WARN(info_log_, "[TwoPhaseWriteManager] GetHandleBounds: invalid handle %d, using handle 6 bounds", handle);
      *lower_bound = 0.0;
      *upper_bound = kCompactionHandle6Threshold;
      break;
  }
}

Status TwoPhaseWriteManager::MoveFileAtomically(const std::string& /*src_path*/,
                                                const std::string& /*dst_path*/,
                                                int /*target_handle*/) {
  // 已废弃：不再使用临时文件，改用内存缓冲区
  // 保留此函数用于兼容，但不应被调用
  ROCKS_LOG_ERROR(info_log_, "[TwoPhaseWriteManager] MoveFileAtomically: 此函数已废弃，不应被调用！");
  return Status::NotSupported("MoveFileAtomically is deprecated, use WriteMemoryBufferToFile instead");
}

Status TwoPhaseWriteManager::WriteMemoryBufferToFile(uint64_t file_number,
                                                      const std::string& buffer_data,
                                                      const std::string& target_path,
                                                      int target_handle) {
  (void)file_number;
  // 使用 RocksDB 的 WritableFileWriter 将内存缓冲区数据写入目标文件
  
  Env* env = Env::Default();
  FileSystem* fs = env->GetFileSystem().get();
  
  // uint64_t buffer_size = buffer_data.size();  // 暂时注释，未使用
  
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
  
  // 创建 WritableFileWriter（和第一次写入完全一样）
  std::unique_ptr<WritableFileWriter> writer(
      new WritableFileWriter(std::move(dst_file), target_path, dst_options,
                            nullptr, nullptr, nullptr, Histograms::SST_WRITE_MICROS,
                            std::vector<std::shared_ptr<EventListener>>(),
                            nullptr, false, false));
  
  // 性能测量：写入数据
  // uint64_t t0_write = Env::Default()->NowMicros();  // 暂时注释，未使用
  // 将内存缓冲区数据写入文件
  Slice data_slice(buffer_data.data(), buffer_data.size());
  IOOptions io_opts;
  io_s = writer->Append(io_opts, data_slice);
  if (!io_s.ok()) {
    return io_s;
  }
  // uint64_t t1_write = Env::Default()->NowMicros();  // 暂时注释，未使用
  // 性能日志已简化
  
  // 性能测量：关闭文件
  // uint64_t t0_close = Env::Default()->NowMicros();  // 暂时注释，未使用
  // 关闭文件（确保文件完全写入并可见）
  // 注意：不调用Sync，Close会确保数据写入到文件系统
  io_s = writer->Close(io_opts);
  if (!io_s.ok()) {
    return io_s;
  }
  // uint64_t t1_close = Env::Default()->NowMicros();  // 暂时注释，未使用
  // 性能日志已简化

  return Status::OK();
}

// 已移除：所有删除队列和移动队列相关函数 - 不再管理文件删除

Status TwoPhaseWriteManager::LoadModels(const std::string& model_dir) {
  (void)model_dir;  // 在函数内部使用，不需要标记为未使用

  // ROCKSDB_ML_COLLECT_ONLY=1：只收集特征不预测，跳过模型加载
  const char* collect_only = getenv("ROCKSDB_ML_COLLECT_ONLY");
  if (collect_only != nullptr && std::string(collect_only) == "1") {
    if (info_log_) {
      ROCKS_LOG_INFO(info_log_,
                     "[TwoPhaseWriteManager][FALLBACK] LoadModels skipped: "
                     "ROCKSDB_ML_COLLECT_ONLY=1 (no model load).");
    }
    return Status::OK();
  }
  // ROCKSDB_ML_PREDICT!=1：未开启模型预测，不加载模型（仅收集特征/二次写入时无需模型）
  const char* ml_predict = getenv("ROCKSDB_ML_PREDICT");
  if (ml_predict == nullptr || std::string(ml_predict) != "1") {
    if (info_log_) {
      ROCKS_LOG_INFO(info_log_,
                     "[TwoPhaseWriteManager][FALLBACK] LoadModels skipped: "
                     "ROCKSDB_ML_PREDICT is not 1 (no ML model load).");
    }
    return Status::OK();
  }

  // 验证模型目录是否存在
  Env* env = Env::Default();
  Status dir_status = env->FileExists(model_dir);
  if (!dir_status.ok()) {
    ROCKS_LOG_ERROR(info_log_, "[TwoPhaseWriteManager::LoadModels] ERROR: Model directory does not exist or is not accessible: %s", model_dir.c_str());
    ROCKS_LOG_ERROR(info_log_, "[TwoPhaseWriteManager::LoadModels] ERROR: dir_status=%s", dir_status.ToString().c_str());
    return Status::NotFound("Model directory does not exist: " + model_dir);
  }
#ifdef ROCKSDB_ML_PREDICT_PYTHON
  if (InitializeMLPredictorByLevel()) {
    return Status::OK();
  } else {
    ROCKS_LOG_ERROR(info_log_, "[TwoPhaseWriteManager::LoadModels] Python ML predictor initialization failed!");
    return Status::Corruption("Python ML predictor initialization failed");
  }
#elif defined(ROCKSDB_ML_PREDICT_ONNX)
  // 使用 ONNX Runtime 预测器（无 GIL 开销，性能好）
  // ONNX 模型目录：环境变量 ROCKSDB_ONNX_MODELS_PATH 或默认路径
  std::string onnx_models_dir;
  const char* onnx_env = getenv("ROCKSDB_ONNX_MODELS_PATH");
  if (onnx_env && strlen(onnx_env) > 0) {
    onnx_models_dir = onnx_env;
  } else {
    // 默认使用 rocksdb 源码目录下的 models_onnx
    onnx_models_dir = "/home/usr/test_environment/rocksdb/models_onnx";
    if (info_log_) {
      static std::atomic<bool> onnx_default_logged{false};
      if (!onnx_default_logged.exchange(true, std::memory_order_relaxed)) {
        ROCKS_LOG_WARN(info_log_,
                       "[TwoPhaseWriteManager][FALLBACK] "
                       "ROCKSDB_ONNX_MODELS_PATH unset; using default dir: %s",
                       onnx_models_dir.c_str());
      }
    }
  }
  
  if (InitializeOnnxPredictor(onnx_models_dir.c_str())) {
    return Status::OK();
  } else {
    ROCKS_LOG_ERROR(info_log_, "[TwoPhaseWriteManager::LoadModels] ONNX predictor initialization failed!");
    return Status::Corruption("ONNX predictor initialization failed");
  }
#else
  if (info_log_) {
    ROCKS_LOG_WARN(info_log_,
                   "[TwoPhaseWriteManager][FALLBACK] LoadModels: no ML backend "
                   "compiled (PYTHON/ONNX); LoadModels is no-op.");
  }
  return Status::OK();
#endif
}

void TwoPhaseWriteManager::LoadFeatureSubset(const std::string& model_dir) {
  feature_subset_indices_.clear();
  n_features_subset_ = 0;
  std::string path = model_dir + "/feature_subset.txt";
  std::ifstream f(path);
  if (!f.is_open()) {
    if (info_log_) {
      ROCKS_LOG_WARN(info_log_,
                     "[TwoPhaseWriteManager][FALLBACK] feature_subset.txt not "
                     "found under %s; using full 69-dim features.",
                     model_dir.c_str());
    }
    return;
  }
  size_t n = 0;
  if (!(f >> n) || n == 0 || n > 69) {
    if (info_log_) {
      ROCKS_LOG_WARN(info_log_, "[TwoPhaseWriteManager] feature_subset.txt invalid n=%zu", n);
    }
    return;
  }
  feature_subset_indices_.reserve(n);
  for (size_t i = 0; i < n; ++i) {
    int idx = -1;
    if (!(f >> idx) || idx < 0 || idx >= 69) {
      feature_subset_indices_.clear();
      n_features_subset_ = 0;
      if (info_log_) {
        ROCKS_LOG_WARN(info_log_, "[TwoPhaseWriteManager] feature_subset.txt invalid index at position %zu", i);
      }
      return;
    }
    feature_subset_indices_.push_back(idx);
  }
  n_features_subset_ = n;
}

double TwoPhaseWriteManager::PredictLifetime(int level,
                                            const std::vector<double>& features) {
  if (features.size() != 69) {
    ROCKS_LOG_ERROR(info_log_, "[TwoPhaseWriteManager] PredictLifetime: 特征数量错误！期望69，实际%zu",
            features.size());
    return -1.0;
  }
  
#ifdef ROCKSDB_ML_PREDICT_PYTHON
  double result = PredictFileLifetimePythonByLevel(features.data(), features.size(), level);
  if (result < 0) {
    ROCKS_LOG_WARN(info_log_, "[TwoPhaseWriteManager][FALLBACK] PredictLifetime: Python预测失败，使用默认值");
    result = 100.0 * level;
    ROCKS_LOG_WARN(info_log_, "[TwoPhaseWriteManager] PredictLifetime:   - 使用默认值: %.2f秒", result);
  }
  
  return result;
#elif defined(ROCKSDB_ML_PREDICT_ONNX)
  // 使用 ONNX Runtime 预测
  double result = PredictFileLifetimeONNX(features.data(), features.size(), level);
  
  if (result < 0) {
    ROCKS_LOG_WARN(info_log_, "[TwoPhaseWriteManager][FALLBACK] PredictLifetime: ONNX预测失败（可能正在关闭），使用默认值");
    ROCKS_LOG_WARN(info_log_, "[TwoPhaseWriteManager] PredictLifetime:   - level: %d", level);
    ROCKS_LOG_WARN(info_log_, "[TwoPhaseWriteManager] PredictLifetime:   - features.size(): %zu", features.size());
    result = 100.0 * level;
    ROCKS_LOG_WARN(info_log_, "[TwoPhaseWriteManager] PredictLifetime:   - 使用默认值: %.2f秒", result);
    return result;
  }
  return result;
#else
  (void)level;
  (void)features;
  ROCKS_LOG_ERROR(info_log_, "[TwoPhaseWriteManager] PredictLifetime: 未定义ML预测后端");
  return -1.0;
#endif
}

Status TwoPhaseWriteManager::CreateMemoryBuffer(uint64_t file_number) {
  std::unique_lock<std::shared_mutex> lock(buffer_mutex_);
  
  // 检查缓冲区是否已存在 - 如果存在，这是不应该发生的情况
  if (memory_buffers_.find(file_number) != memory_buffers_.end()) {
    ROCKS_LOG_FATAL(info_log_, "[TwoPhaseWriteManager] CreateMemoryBuffer: file #%" PRIu64 " - 缓冲区已存在！这是不应该发生的情况！", file_number);
    abort();  // FAIL IMMEDIATELY - NO ERROR TOLERANCE
  }
  
  memory_buffers_[file_number] = std::string();
  memory_buffers_[file_number].reserve(48 * 1024 * 1024);  // 预分配48MB
  
  return Status::OK();
}

Status TwoPhaseWriteManager::AppendToMemoryBuffer(uint64_t file_number, const Slice& data) {
  std::unique_lock<std::shared_mutex> lock(buffer_mutex_);
  auto it = memory_buffers_.find(file_number);
  if (it == memory_buffers_.end()) {
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
  auto it = memory_buffers_.find(file_number);
  if (it == memory_buffers_.end()) {
    return Status::NotFound("Memory buffer not found for file " + std::to_string(file_number));
  }
  *data = it->second;
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
  
#ifdef ROCKSDB_ML_PREDICT_ONNX
  // Explicitly shutdown ONNX predictor when TwoPhaseWriteManager is shutdown.
  // This ensures ONNX is shutdown only after all background tasks are done.
  ShutdownOnnxPredictor();
#endif
}

int TwoPhaseWriteManager::PredictHandleBeforeWrite(
    uint64_t file_number, int level,
    const Slice& first_key,
    void* sub_compact_ptr,
    void* cfd_ptr,
    void* db_mutex_ptr) {
  
  if (!initialized_.load() || !enable_phase2_) {
    if (info_log_) {
      static std::atomic<bool> ph2_off_logged{false};
      if (!ph2_off_logged.exchange(true, std::memory_order_relaxed)) {
        ROCKS_LOG_WARN(info_log_,
                       "[PredictHandleBeforeWrite][FALLBACK] Phase2 off or "
                       "manager not initialized; returning -1 (caller uses "
                       "non-ML handle path).");
      }
    }
    return -1;
  }
  
  // 类型转换
  SubcompactionState* sub_compact = static_cast<SubcompactionState*>(sub_compact_ptr);
  ColumnFamilyData* cfd = static_cast<ColumnFamilyData*>(cfd_ptr);
  InstrumentedMutex* db_mutex = static_cast<InstrumentedMutex*>(db_mutex_ptr);
  
  if (!sub_compact || !cfd || !db_mutex) {
    ROCKS_LOG_WARN(info_log_, "[PredictHandleBeforeWrite] file #%" PRIu64 
        " - null pointer: sub_compact=%p, cfd=%p, db_mutex=%p",
        file_number, sub_compact, cfd, db_mutex);
    return -1;
  }
  
  // 获取 compaction 信息
  const Compaction* compaction = sub_compact->compaction;
  if (!compaction) {
    ROCKS_LOG_WARN(info_log_, "[PredictHandleBeforeWrite] file #%" PRIu64 " - null compaction", file_number);
    return -1;
  }
  
  // 精确的 smallest key（从 first_key 参数获取）
  InternalKey smallest_key;
  smallest_key.DecodeFrom(first_key);
  
  // 估算 largest key
  // 方法：总 range 除以 estimated_output_files，按 current_file_index 均分
  InternalKey estimated_largest;

  // 获取总 range 边界
  Slice range_start = sub_compact->start.has_value()
                          ? *(sub_compact->start)
                          : compaction->GetSmallestUserKey();
  Slice range_end = sub_compact->end.has_value()
                        ? *(sub_compact->end)
                        : compaction->GetLargestUserKey();

  uint64_t target_file_size = compaction->target_output_file_size();
  uint64_t total_input_size = compaction->CalculateTotalInputSize();
  int estimated_output_files = std::max(1, static_cast<int>(
      (total_input_size + target_file_size - 1) / target_file_size));
  int current_file_index =
      static_cast<int>(sub_compact->Current().GetOutputs().size());

  std::string estimated_ub = EstimateFileLargestKey(
      range_start, range_end, estimated_output_files, current_file_index);
  if (!estimated_ub.empty()) {
    estimated_largest.SetMinPossibleForUserKey(estimated_ub);
  } else {
    estimated_largest.SetMinPossibleForUserKey(range_end);
  }
  
  // 使用 25 维特征子集进行特征收集与 ML 预测，决定 handle
  // 使用精确的 smallest 和估算的 largest 计算特征
  // 使用 target_file_size 作为估算的文件大小
  MLFeatures features;
  
  // 记录特征计算时间
  uint64_t t_feature_start = Env::Default()->NowMicros();
  
  bool features_ok = CalculateMLFeaturesBeforeWrite(
      smallest_key,
      estimated_largest,
      0,  // num_entries 未知，使用 0
      target_file_size,  // 使用 target_file_size 作为估算的文件大小
      level,
      cfd,
      &features,
      db_mutex,
      (n_features_subset_ > 0 && !feature_subset_indices_.empty()) ? &feature_subset_indices_ : nullptr);
  
  uint64_t t_feature_end = Env::Default()->NowMicros();
  uint64_t feature_calc_time = t_feature_end - t_feature_start;
  
  if (!features_ok) {
    int fh = 7 + (level - 1);
    ROCKS_LOG_WARN(info_log_,
                   "[PredictHandleBeforeWrite][FALLBACK] file #%" PRIu64
                   " level=%d feature calculation failed; using level-based "
                   "handle %d",
                   file_number, level, fh);
    return fh;
  }
  
  // 将特征转换为数组（始终算满 69 维，再按子集取或全传）
  std::vector<double> feature_array(69);
  MLFeaturesToArray(features, feature_array.data(), feature_array.size());
  
  // 预测生命周期
  double predicted_lifetime = -1.0;
  
  // 记录预测时间
  uint64_t t_predict_start = Env::Default()->NowMicros();
  
#ifdef ROCKSDB_ML_PREDICT_PYTHON
  if (n_features_subset_ > 0 && feature_subset_indices_.size() == n_features_subset_) {
    std::vector<double> subset_array(n_features_subset_);
    for (size_t i = 0; i < n_features_subset_; ++i) {
      int idx = feature_subset_indices_[i];
      subset_array[i] = (idx >= 0 && idx < 69) ? feature_array[static_cast<size_t>(idx)] : 0.0;
    }
    predicted_lifetime = PredictFileLifetimePythonByLevel(
        subset_array.data(), n_features_subset_, level);
  } else {
    predicted_lifetime = PredictFileLifetimePythonByLevel(
        feature_array.data(), feature_array.size(), level);
  }
#elif defined(ROCKSDB_ML_PREDICT_ONNX)
  predicted_lifetime = PredictFileLifetimeOnnx(level, feature_array);
#endif

  uint64_t t_predict_end = Env::Default()->NowMicros();
  uint64_t python_predict_time = t_predict_end - t_predict_start;
  
  // 输出细分时间
  ROCKS_LOG_INFO(info_log_, 
      "[TIMING_DETAIL] file #%" PRIu64 " FeatureCalc=%" PRIu64 "us, PythonPredict=%" PRIu64 "us",
      file_number, feature_calc_time, python_predict_time);

  // 累计开销（用于 Shutdown 时输出 [ML_OVERHEAD_SUMMARY]）
  total_feature_calc_us_.fetch_add(feature_calc_time);
  total_python_predict_us_.fetch_add(python_predict_time);
  predict_call_count_.fetch_add(1);
  
  if (predicted_lifetime < 0) {
    int default_handle = 7 + (level - 1);
    ROCKS_LOG_WARN(info_log_,
                   "[PredictHandleBeforeWrite][FALLBACK] file #%" PRIu64
                   " level=%d prediction failed (pred=%.2f); using "
                   "level-based handle %d",
                   file_number, level, predicted_lifetime, default_handle);
    ROCKS_LOG_INFO(info_log_,
                   "[FILE_HANDLE] file #%" PRIu64 " level=%d -> handle %d "
                   "(predict_failed, default)",
                   file_number, level, default_handle);
    return default_handle;
  }
  
  // 映射生命周期到 handle
  int target_handle = MapLifetimeToHandle(predicted_lifetime, level);
  
  ROCKS_LOG_INFO(info_log_,
                 "[FILE_HANDLE] file #%" PRIu64 " level=%d -> handle %d (predict, lifetime=%.2fs)",
                 file_number, level, target_handle, predicted_lifetime);
  return target_handle;
}

void TwoPhaseWriteManager::CollectAndPrintFeaturesBeforeWrite(
    uint64_t file_number, int level,
    const Slice& first_key,
    void* sub_compact_ptr,
    void* cfd_ptr,
    void* db_mutex_ptr) {
  if (!initialized_.load() || !enable_phase2_) {
    return;
  }

  SubcompactionState* sub_compact = static_cast<SubcompactionState*>(sub_compact_ptr);
  ColumnFamilyData* cfd = static_cast<ColumnFamilyData*>(cfd_ptr);
  InstrumentedMutex* db_mutex = static_cast<InstrumentedMutex*>(db_mutex_ptr);

  if (!sub_compact || !cfd || !db_mutex) {
    ROCKS_LOG_WARN(info_log_, "[CollectAndPrintFeatures] file #%" PRIu64
        " - null pointer, skip",
        file_number);
    return;
  }

  const Compaction* compaction = sub_compact->compaction;
  if (!compaction) {
    ROCKS_LOG_WARN(info_log_, "[CollectAndPrintFeatures] file #%" PRIu64 " - null compaction", file_number);
    return;
  }

  InternalKey smallest_key;
  smallest_key.DecodeFrom(first_key);

  InternalKey estimated_largest;
  Slice range_start = sub_compact->start.has_value()
                          ? *(sub_compact->start)
                          : compaction->GetSmallestUserKey();
  Slice range_end = sub_compact->end.has_value()
                        ? *(sub_compact->end)
                        : compaction->GetLargestUserKey();

  uint64_t target_file_size = compaction->target_output_file_size();
  uint64_t total_input_size = compaction->CalculateTotalInputSize();
  int estimated_output_files = std::max(1, static_cast<int>(
      (total_input_size + target_file_size - 1) / target_file_size));
  int current_file_index =
      static_cast<int>(sub_compact->Current().GetOutputs().size());

  std::string estimated_ub = EstimateFileLargestKey(
      range_start, range_end, estimated_output_files, current_file_index);
  if (!estimated_ub.empty()) {
    estimated_largest.SetMinPossibleForUserKey(estimated_ub);
  } else {
    estimated_largest.SetMinPossibleForUserKey(range_end);
  }

  MLFeatures features;
  bool features_ok = CalculateMLFeaturesBeforeWrite(
      smallest_key,
      estimated_largest,
      0,
      target_file_size,
      level,
      cfd,
      &features,
      db_mutex,
      nullptr);  // 计算全部 69 维特征

  if (!features_ok) {
    ROCKS_LOG_WARN(info_log_, "[CollectAndPrintFeatures] file #%" PRIu64
        " level=%d - feature calculation failed",
        file_number, level);
    return;
  }

  std::vector<double> feature_array(69);
  MLFeaturesToArray(features, feature_array.data(), feature_array.size());

  // 打印 69 维特征（紧凑格式）
  std::string buf;
  for (size_t i = 0; i < feature_array.size(); ++i) {
    if (i > 0) buf += ",";
    char tmp[32];
    snprintf(tmp, sizeof(tmp), "%.6g", feature_array[i]);
    buf += tmp;
  }
  ROCKS_LOG_INFO(info_log_, "[ML_FEATURES] file #%" PRIu64 " level=%d 69dims=[%s]",
                 file_number, level, buf.c_str());
}

}  // namespace ROCKSDB_NAMESPACE

