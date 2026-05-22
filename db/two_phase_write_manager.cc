//  Copyright (c) 2011-present, Facebook, Inc.  All rights reserved.
//  This source code is licensed under both the GPLv2 (found in the
//  COPYING file in the root directory) and Apache 2.0 License
//  (found in the LICENSE.Apache file in the root directory).

#include "db/two_phase_write_manager.h"

#include "rocksdb/io_status.h"

#include <algorithm>
#include <atomic>
#include <cctype>
#include <cmath>
#include <cerrno>
#include <cinttypes>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <fcntl.h>
#include <cstring>
#include <linux/fs.h>
#if defined(__linux__)
#include <endian.h>
#include <linux/nvme_ioctl.h>
#include <sys/ioctl.h>
#include <sys/sysmacros.h>
#endif
#include <errno.h>
#include <stdlib.h>  // for posix_memalign
#include <thread>    // for std::this_thread::sleep_for
#include <chrono>    // for std::chrono::milliseconds
#include <limits>
#include <climits>
#include <mutex>     // std::call_once, std::mutex, std::lock_guard
#include <string>
#include <unordered_map>  // for thread_local std::unordered_map
#include <vector>

#include "rocksdb/env.h"
#include "rocksdb/status.h"
#include "rocksdb/slice.h"
#include "rocksdb/statistics.h"
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

std::atomic<uint64_t> g_adaptive_compaction_jobs_started_total{0};
std::atomic<bool> g_adaptive_compaction_counting_for_probe{false};

void TwoPhaseAdaptiveNoteCompactionJobStarted() {
  if (!g_adaptive_compaction_counting_for_probe.load(std::memory_order_relaxed)) {
    return;
  }
  g_adaptive_compaction_jobs_started_total.fetch_add(1, std::memory_order_relaxed);
}

namespace {
void EnsureLevelBudgetsFromEnv(Statistics* db_statistics_for_adaptive = nullptr,
                               const char* adaptive_fdp_db_path = nullptr);
void LogBudgetConfigToInfoIfAny(Logger* info_log);

// no-fdp：所有 level（含 L0）统一写入的 FDP handle（与原先 L0 使用的 handle 6 一致）
constexpr int kNoFdpUnifiedHandle = 6;

// native-base（Optimized classification）：SST 侧整型值 = WriteLifeTimeHint
// SHORT=WAL 仍由 DBImpl::CalculateWALWriteHint；SST：MEDIUM=L0–L3，LONG=L4，EXTREME=≥L5
int NativeCompactionMetadataHandle(int level) {
  if (level <= 3) {
    return 3;  // MEDIUM：L0 ~ L3
  }
  if (level == 4) {
    return 4;  // LONG
  }
  if (level >= 5) {
    return 5;  // EXTREME：≥ L5
  }
  return 3;
}
}  // namespace

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
                                        Logger* info_log,
                                        Statistics* db_statistics) {
  
  if (initialized_.load()) {
    return Status::OK();
  }

  model_dir_ = model_dir;
  db_path_ = db_path;
  enable_phase2_ = enable_phase2;
  info_log_ = info_log;
  handle_write_policy_ = HandleWritePolicy::kLevelBase;

  if (enable_phase2_) {
    const char* hash_env = std::getenv("ROCKSDB_HASH_HANDLE");
    if (hash_env == nullptr || hash_env[0] == '\0') {
      return Status::InvalidArgument(
          "ROCKSDB_ENABLE_PHASE2=1 requires ROCKSDB_HASH_HANDLE to be set "
          "to \"0\" (level-base), \"1\" (hash-base), \"2\" or \"no-fdp\" "
          "(single handle for all levels), or \"3\"/\"native\"/\"native-base\" "
          "(optimized tier: MEDIUM=L0–L3, LONG=L4, EXTREME≥L5; WAL=SHORT). "
          "Unset or empty is an error (no fallback).");
    }
    if (std::strcmp(hash_env, "0") == 0) {
      handle_write_policy_ = HandleWritePolicy::kLevelBase;
    } else if (std::strcmp(hash_env, "1") == 0) {
      handle_write_policy_ = HandleWritePolicy::kHashBase;
    } else if (std::strcmp(hash_env, "2") == 0 ||
               std::strcmp(hash_env, "no-fdp") == 0) {
      handle_write_policy_ = HandleWritePolicy::kNoFdp;
    } else if (std::strcmp(hash_env, "3") == 0 ||
               std::strcmp(hash_env, "native") == 0 ||
               std::strcmp(hash_env, "native-base") == 0) {
      handle_write_policy_ = HandleWritePolicy::kNativeBase;
    } else {
      return Status::InvalidArgument(
          "ROCKSDB_ENABLE_PHASE2=1 requires ROCKSDB_HASH_HANDLE to be "
          "\"0\" (level-base), \"1\" (hash-base), \"2\" or \"no-fdp\" "
          "(all levels -> single handle; currently handle 6), or "
          "\"3\"/\"native\"/\"native-base\" (optimized classification hints). "
          "Any other value is an error (no fallback).");
    }
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
      const char* policy_str = "0 (level-base)";
      switch (handle_write_policy_) {
        case HandleWritePolicy::kLevelBase:
          policy_str = "0 (level-base)";
          break;
        case HandleWritePolicy::kHashBase:
          policy_str = "1 (hash-base)";
          break;
        case HandleWritePolicy::kNoFdp:
          policy_str = "no-fdp (all levels -> handle 6)";
          break;
        case HandleWritePolicy::kNativeBase:
          policy_str = "3/native-base (opt: L0–3→MED, L4→LONG, ≥L5→EXT)";
          break;
      }
      ROCKS_LOG_INFO(info_log_,
                     "[TwoPhaseWriteManager] initialized enable_phase2=1 "
                     "ROCKSDB_HASH_HANDLE=%s",
                     policy_str);
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

  EnsureLevelBudgetsFromEnv(db_statistics, db_path_.c_str());
  LogBudgetConfigToInfoIfAny(info_log_);

  initialized_.store(true);
  return Status::OK();
}

bool TwoPhaseWriteManager::IsNativeBasePolicy() const {
  return handle_write_policy_ == HandleWritePolicy::kNativeBase;
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
  if (handle_write_policy_ == HandleWritePolicy::kNoFdp) {
    // 不按寿命分 handle：全 level 单 handle，跳过 ML 预测
    (void)features;
    predicted_lifetime = 0.0;
    target_handle = kNoFdpUnifiedHandle;
  } else if (handle_write_policy_ == HandleWritePolicy::kNativeBase) {
    (void)features;
    predicted_lifetime = 0.0;
    target_handle = NativeCompactionMetadataHandle(level);
  } else {
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
  }
#else
  // 无 ML 预测：直接按 level 映射 handle（仅用于自定义 compaction 的元数据，不二次写入）
  (void)features;
  predicted_lifetime = 0.0;
  if (handle_write_policy_ == HandleWritePolicy::kNoFdp) {
    target_handle = kNoFdpUnifiedHandle;
  } else if (handle_write_policy_ == HandleWritePolicy::kNativeBase) {
    target_handle = NativeCompactionMetadataHandle(level);
  } else {
    target_handle = level + 6;
  }
#endif

  const bool handle_valid =
      (target_handle >= 6 && target_handle <= 12) ||
      (handle_write_policy_ == HandleWritePolicy::kNativeBase && target_handle >= 3 &&
       target_handle <= 5);
  if (!handle_valid) {
    ROCKS_LOG_ERROR(info_log_, "[TwoPhaseWriteManager] CollectFeaturesAndPredict: file #%" PRIu64 " - handle=%d 无效！level=%d, predicted_lifetime=%.2f",
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

  // too-far 用的 compaction 年龄阈值（秒）：handle 7–11，可由环境变量覆盖（见
  // ROCKSDB_TOO_FAR_THRESHOLDS_SEC）；默认与上列 constexpr 一致。
  double g_compaction_threshold_7_to_11[5] = {
      kCompactionHandle7Threshold, kCompactionHandle8Threshold,
      kCompactionHandle9Threshold, kCompactionHandle10Threshold,
      kCompactionHandle11Threshold};
  std::once_flag g_compaction_threshold_once;
  bool g_compaction_threshold_from_env = false;
  std::string g_compaction_threshold_env_raw("(unset)");
  std::atomic<bool> g_compaction_threshold_stderr_done{false};

  void LogCompactionThresholdsToStderrOnce() {
    if (g_compaction_threshold_stderr_done.exchange(true)) {
      return;
    }
    fprintf(stderr,
            "[TwoPhaseWriteManager] too_far thresholds (sec) h7-h11: "
            "%.3f %.3f %.3f %.3f %.3f  source=%s  raw=%s\n",
            g_compaction_threshold_7_to_11[0], g_compaction_threshold_7_to_11[1],
            g_compaction_threshold_7_to_11[2], g_compaction_threshold_7_to_11[3],
            g_compaction_threshold_7_to_11[4],
            g_compaction_threshold_from_env ? "env" : "default",
            g_compaction_threshold_env_raw.c_str());
  }

  bool ParseTooFarThresholdsEnv(const char* v, double out[5],
                                 std::string* raw_out) {
    if (v == nullptr || v[0] == '\0') {
      return false;
    }
    raw_out->assign(v);
    const char* p = v;
    for (int i = 0; i < 5; i++) {
      char* end = nullptr;
      errno = 0;
      double x = strtod(p, &end);
      if (end == p || errno == ERANGE) {
        return false;
      }
      if (std::isnan(x) || std::isinf(x)) {
        return false;
      }
      if (x < 0.0) {
        return false;
      }
      out[i] = x;
      p = end;
      if (i < 4) {
        while (*p != '\0' &&
               std::isspace(static_cast<unsigned char>(*p))) {
          p++;
        }
        if (*p != ',') {
          return false;
        }
        p++;
        while (*p != '\0' &&
               std::isspace(static_cast<unsigned char>(*p))) {
          p++;
        }
      } else {
        while (*p != '\0' &&
               std::isspace(static_cast<unsigned char>(*p))) {
          p++;
        }
        if (*p != '\0') {
          return false;
        }
      }
    }
    // 不要求单调递增：五元组按 handle7..11 独立生效，便于从各 level 的 p50 等统计直接填入。
    return true;
  }

  void EnsureCompactionThresholdsFromEnv() {
    std::call_once(g_compaction_threshold_once, []() {
      g_compaction_threshold_7_to_11[0] = kCompactionHandle7Threshold;
      g_compaction_threshold_7_to_11[1] = kCompactionHandle8Threshold;
      g_compaction_threshold_7_to_11[2] = kCompactionHandle9Threshold;
      g_compaction_threshold_7_to_11[3] = kCompactionHandle10Threshold;
      g_compaction_threshold_7_to_11[4] = kCompactionHandle11Threshold;
      g_compaction_threshold_from_env = false;

      const char* v = std::getenv("ROCKSDB_TOO_FAR_THRESHOLDS_SEC");
      if (v != nullptr && v[0] != '\0') {
        double parsed[5];
        std::string raw_tmp;
        if (ParseTooFarThresholdsEnv(v, parsed, &raw_tmp)) {
          for (int i = 0; i < 5; i++) {
            g_compaction_threshold_7_to_11[i] = parsed[i];
          }
          g_compaction_threshold_from_env = true;
          g_compaction_threshold_env_raw = std::move(raw_tmp);
        } else {
          fprintf(stderr,
                  "[TwoPhaseWriteManager][WARN] "
                  "ROCKSDB_TOO_FAR_THRESHOLDS_SEC invalid (need 5 non-negative "
                  "comma-separated floats); using defaults "
                  "5,10,15,60,100 (sec). raw=%s\n",
                  v);
          g_compaction_threshold_env_raw.assign(v);
        }
      } else {
        g_compaction_threshold_env_raw.assign("(unset)");
      }
      LogCompactionThresholdsToStderrOnce();
    });
  }

  // 全局一份 K/N：L1–L5 的 compaction pick 与 custom 预约共用同一滑动窗口。
  struct TooFarBudgetState {
    int allow = 0;          // K：本 cycle 允许的 custom 次数；window>0 且 allow==0 表示 0 次配额
    int window = 0;         // N：cycle 窗口大小（>0 才有效；window<=0 或 allow<0 视作关闭 cap）
    std::mutex mu;
    int attempts = 0;       // 本 cycle 已发生 compaction 总次数（custom + baseline）
    int customs_used = 0;   // 本 cycle 已用 custom 次数
    int slot_idx = -1;      // 时间调度档索引（ROCKSDB_TOO_FAR_BUDGET_SCHEDULE）；-1 未绑定
  };

  struct BudgetScheduleSlot {
    int allow = 0;
    int window = 0;
  };

  std::vector<BudgetScheduleSlot> g_budget_schedule;
  double g_budget_schedule_interval_sec = 12.0;
  std::chrono::steady_clock::time_point g_budget_schedule_start{};
  bool g_budget_schedule_enabled = false;
  std::string g_budget_schedule_env_raw("(unset)");

  TooFarBudgetState g_too_far_budget;
  std::once_flag g_level_budgets_once;
  bool g_level_budgets_from_env = false;
  std::string g_level_budgets_env_raw("(unset)");
  std::atomic<bool> g_level_budgets_stderr_done{false};

  void LogLevelBudgetsToStderrOnce() {
    if (g_level_budgets_stderr_done.exchange(true)) {
      return;
    }
    fprintf(stderr,
            "[TwoPhaseWriteManager] too_far budget (global K/N, all levels): "
            "%d/%d  source=%s  raw=%s\n",
            g_too_far_budget.allow, g_too_far_budget.window,
            g_level_budgets_from_env ? "env" : "default",
            g_level_budgets_env_raw.c_str());
  }

  // 解析单段："N" 表示 K=1,N=N（向后兼容）；"K/N" 直接给出 K 和 N。
  // 段以 [s, end) 表示；允许首尾空格；要求 1<=K<=N<=10000。成功返回 true。
  bool ParseSingleKN(const char* s, const char* end, int* k_out, int* n_out) {
    while (s < end && std::isspace(static_cast<unsigned char>(*s))) s++;
    while (end > s && std::isspace(static_cast<unsigned char>(*(end - 1)))) end--;
    if (s == end) return false;
    // 首先尝试在段内找 '/'
    const char* slash = nullptr;
    for (const char* q = s; q < end; ++q) {
      if (*q == '/') { slash = q; break; }
    }
    auto parse_int = [](const char* a, const char* b, long* out) -> bool {
      if (a == b) return false;
      // 复制到栈缓冲再 strtol（避免依赖 *b 是 '\0'）
      char buf[32];
      size_t n = static_cast<size_t>(b - a);
      if (n == 0 || n >= sizeof(buf)) return false;
      memcpy(buf, a, n);
      buf[n] = '\0';
      // 不允许内部空格（前面已 trim 段两端）
      for (size_t i = 0; i < n; i++) {
        if (std::isspace(static_cast<unsigned char>(buf[i]))) return false;
      }
      char* tail = nullptr;
      errno = 0;
      long x = std::strtol(buf, &tail, 10);
      if (tail == buf || *tail != '\0' || errno == ERANGE) return false;
      *out = x;
      return true;
    };
    long n_val = 0;
    long k_val = 1;
    if (slash == nullptr) {
      if (!parse_int(s, end, &n_val)) return false;
    } else {
      if (!parse_int(s, slash, &k_val)) return false;
      if (!parse_int(slash + 1, end, &n_val)) return false;
    }
    if (k_val < 1 || n_val < 1 || k_val > n_val || n_val > 10000) {
      return false;
    }
    *k_out = static_cast<int>(k_val);
    *n_out = static_cast<int>(n_val);
    return true;
  }

  // 与 ParseSingleKN 相同，但允许 K=0（仅用于 ADAPTIVE PROBE_KN 列表）。
  bool ParseSingleKNAdaptiveAllowZero(const char* s, const char* end, int* k_out,
                                      int* n_out) {
    while (s < end && std::isspace(static_cast<unsigned char>(*s))) s++;
    while (end > s && std::isspace(static_cast<unsigned char>(*(end - 1)))) end--;
    if (s == end) return false;
    const char* slash = nullptr;
    for (const char* q = s; q < end; ++q) {
      if (*q == '/') {
        slash = q;
        break;
      }
    }
    auto parse_int = [](const char* a, const char* b, long* out) -> bool {
      if (a == b) return false;
      char buf[32];
      size_t n = static_cast<size_t>(b - a);
      if (n == 0 || n >= sizeof(buf)) return false;
      memcpy(buf, a, n);
      buf[n] = '\0';
      for (size_t i = 0; i < n; i++) {
        if (std::isspace(static_cast<unsigned char>(buf[i]))) return false;
      }
      char* tail = nullptr;
      errno = 0;
      long x = std::strtol(buf, &tail, 10);
      if (tail == buf || *tail != '\0' || errno == ERANGE) return false;
      *out = x;
      return true;
    };
    long n_val = 0;
    long k_val = 1;
    if (slash == nullptr) {
      if (!parse_int(s, end, &n_val)) return false;
    } else {
      if (!parse_int(s, slash, &k_val)) return false;
      if (!parse_int(slash + 1, end, &n_val)) return false;
    }
    if (k_val < 0 || n_val < 1 || k_val > n_val || n_val > 10000) {
      return false;
    }
    *k_out = static_cast<int>(k_val);
    *n_out = static_cast<int>(n_val);
    return true;
  }

  bool ParseAdaptiveProbeSlotsEnv(const char* v,
                                    std::vector<BudgetScheduleSlot>* out) {
    out->clear();
    if (v == nullptr || v[0] == '\0') {
      return false;
    }
    const char* seg_start = v;
    const char* end_all = v + std::strlen(v);
    for (const char* q = v; q <= end_all; ++q) {
      if (q == end_all || *q == ',') {
        int k = 0, n = 0;
        if (!ParseSingleKNAdaptiveAllowZero(seg_start, q, &k, &n)) {
          out->clear();
          return false;
        }
        out->push_back(BudgetScheduleSlot{k, n});
        seg_start = q + 1;
        if (q == end_all) {
          break;
        }
      }
    }
    return !out->empty();
  }

  bool IsTriple0510ProbeSlots(const std::vector<BudgetScheduleSlot>& slots) {
    return slots.size() == 3 && slots[0].allow == 0 && slots[0].window == 10 &&
           slots[1].allow == 5 && slots[1].window == 10 &&
           slots[2].allow == 10 && slots[2].window == 10;
  }

  // ROCKSDB_TOO_FAR_BUDGET 解析：
  //   "N" 或 "K/N"：单组 K/N（全局 cap）
  //   五段逗号分隔（历史 per-level 写法）：仍解析五段，但全局 cap 只采用第一段（L1）；
  //     若五段不完全相同，stderr 警告一次。
  // 任何段不合法 → 整体失败（cap 关闭）。
  bool ParseLevelBudgetsEnv(const char* v, int allow_out[5], int window_out[5],
                            std::string* raw_out) {
    if (v == nullptr || v[0] == '\0') {
      return false;
    }
    raw_out->assign(v);
    const char* p = v;
    const char* end_all = v + std::strlen(v);
    // 计数 ',' 决定单广播还是五段
    int comma_count = 0;
    for (const char* q = p; *q != '\0'; ++q) {
      if (*q == ',') comma_count++;
    }
    if (comma_count == 0) {
      int k = 0, n = 0;
      if (!ParseSingleKN(p, end_all, &k, &n)) return false;
      for (int i = 0; i < 5; i++) {
        allow_out[i] = k;
        window_out[i] = n;
      }
      return true;
    }
    if (comma_count != 4) {
      return false;
    }
    const char* seg_start = p;
    int idx = 0;
    for (const char* q = p; q <= end_all; ++q) {
      if (q == end_all || *q == ',') {
        int k = 0, n = 0;
        if (!ParseSingleKN(seg_start, q, &k, &n)) return false;
        if (idx >= 5) return false;
        allow_out[idx] = k;
        window_out[idx] = n;
        idx++;
        seg_start = q + 1;
        if (q == end_all) break;
      }
    }
    return idx == 5;
  }

  // 逗号分隔多段 K/N，段数任意（>=1）；任一段非法则整体失败。
  bool ParseBudgetScheduleEnv(const char* v, std::vector<BudgetScheduleSlot>* out) {
    out->clear();
    if (v == nullptr || v[0] == '\0') {
      return false;
    }
    const char* seg_start = v;
    const char* end_all = v + std::strlen(v);
    for (const char* q = v; q <= end_all; ++q) {
      if (q == end_all || *q == ',') {
        int k = 0, n = 0;
        if (!ParseSingleKN(seg_start, q, &k, &n)) {
          out->clear();
          return false;
        }
        out->push_back(BudgetScheduleSlot{k, n});
        seg_start = q + 1;
        if (q == end_all) {
          break;
        }
      }
    }
    return !out->empty();
  }

  std::atomic<bool> g_adaptive_enabled{false};
  std::vector<BudgetScheduleSlot> g_adaptive_probe_slots;
  int g_adaptive_probe_slot_sec = 20;
  int g_adaptive_exploit_sec = 180;
  uint64_t g_adaptive_min_bytes = 0;
  std::string g_adaptive_probe_kn_raw("(default)");
  std::string g_adaptive_env_note("(off)");

  enum class AdaptiveBudgetPhase { kProbe, kExploit };
  AdaptiveBudgetPhase g_adaptive_phase = AdaptiveBudgetPhase::kProbe;
  int g_adaptive_probe_index = 0;
  std::chrono::steady_clock::time_point g_adaptive_segment_start{};
  // 探针段起点快照：Statistics::BYTES_WRITTEN 与 g_adaptive_total_user_bytes 均为
  // 「自 DB 创建以来的累计值」；段末用 ReadAdaptiveProbeByteCounter() 减本字段得本段 delta。
  uint64_t g_adaptive_segment_byte_start = 0;
  std::chrono::steady_clock::time_point g_adaptive_exploit_until{};
  std::vector<double> g_adaptive_slot_bps;  // one entry per completed probe slot
  int g_adaptive_round_seq = 0;

  std::atomic<uint64_t> g_adaptive_total_user_bytes{0};
  // 非空：探针用 Statistics::BYTES_WRITTEN；空：用 g_adaptive_total_user_bytes。
  std::atomic<Statistics*> g_adaptive_probe_byte_statistics{nullptr};

  // 返回单调累计计数（非本段字节数）；本段写入量须在调用方做 end - segment_byte_start。
  uint64_t ReadAdaptiveProbeByteCounter() {
    Statistics* st =
        g_adaptive_probe_byte_statistics.load(std::memory_order_relaxed);
    if (st != nullptr) {
      return st->getTickerCount(BYTES_WRITTEN);
    }
    return g_adaptive_total_user_bytes.load(std::memory_order_relaxed);
  }

  // PROBE_SHRINK=1：利用期结束后下一轮探针不扫满档，只在「上轮胜者」附近的 K/N 档上探针。
  bool g_adaptive_probe_shrink_enabled = false;
  // 当前探针轮要依次探测的槽位 = g_adaptive_probe_slots 的下标（全局顺序，与 PROBE_KN 一致）。
  std::vector<int> g_adaptive_probe_slot_plan;
  int g_adaptive_last_win_global = -1;
  int g_adaptive_win_streak = 0;
  int g_adaptive_prev_win_for_streak = -1;
  // 连续若干轮「胜者 == 当轮 plan 最左档」：仅 1 轮则下轮起点锁 plan_min；≥2 则起点再左移一档。
  int g_adaptive_min_plan_win_streak = 0;
  int g_adaptive_last_closed_plan_min_global = -1;
  bool g_adaptive_triple_0510_mode = false;
  std::vector<int> g_adaptive_prev_round_plan;
  // >=0：临时调试，每轮探针结束强制该全局槽胜出；探针计划缩为仅该档。
  int g_adaptive_force_win_global = -1;

  enum class AdaptiveProbeAdvanceMode { kWallClock, kCompactions };
  AdaptiveProbeAdvanceMode g_adaptive_probe_advance_mode =
      AdaptiveProbeAdvanceMode::kWallClock;
  int g_adaptive_probe_slot_compactions = 30;

  enum class AdaptiveProbeMetric { kBps, kWaProduct };
  AdaptiveProbeMetric g_adaptive_probe_metric = AdaptiveProbeMetric::kBps;
  std::string g_adaptive_fdp_stats_file_path;
  std::string g_adaptive_fdp_nvme_device;

  uint64_t g_adaptive_segment_compaction_start = 0;
  uint64_t g_adaptive_seg_user_b = 0;
  uint64_t g_adaptive_seg_flush_b = 0;
  uint64_t g_adaptive_seg_compact_b = 0;
  // FDP HBMW/MBMW 各 128 位（小端 lo/hi）；无 NVMe 时文件路径仅填 lo，hi=0。
  uint64_t g_adaptive_seg_fdp_h0 = 0;
  uint64_t g_adaptive_seg_fdp_h1 = 0;
  uint64_t g_adaptive_seg_fdp_m0 = 0;
  uint64_t g_adaptive_seg_fdp_m1 = 0;

  std::vector<double> g_adaptive_slot_app_wa;
  std::vector<double> g_adaptive_slot_dev_wa;

  bool FdpWaSourceEnabled() {
    return !g_adaptive_fdp_nvme_device.empty() ||
           !g_adaptive_fdp_stats_file_path.empty();
  }

#if defined(__linux__) && defined(__SIZEOF_INT128__) && defined(__GNUC__)
  using FdpU128 = unsigned __int128;
  static FdpU128 FdpU128FromPair(uint64_t lo, uint64_t hi) {
    return (static_cast<FdpU128>(hi) << 64) | static_cast<FdpU128>(lo);
  }
  static void FdpU128Sub(uint64_t a0, uint64_t a1, uint64_t b0, uint64_t b1,
                        uint64_t* o0, uint64_t* o1) {
    FdpU128 a = FdpU128FromPair(a0, a1);
    FdpU128 b = FdpU128FromPair(b0, b1);
    if (a >= b) {
      FdpU128 d = a - b;
      *o0 = static_cast<uint64_t>(d);
      *o1 = static_cast<uint64_t>(d >> 64);
    } else {
      *o0 = *o1 = 0;
    }
  }
#else
  static void FdpU128Sub(uint64_t a0, uint64_t a1, uint64_t b0, uint64_t b1,
                        uint64_t* o0, uint64_t* o1) {
    (void)a1;
    (void)b1;
    *o0 = a0 >= b0 ? a0 - b0 : 0;
    *o1 = 0;
  }
#endif

  bool ReadFdpHbmwMbmwFromFile(uint64_t* hbmw, uint64_t* mbmw) {
    *hbmw = *mbmw = 0;
    if (g_adaptive_fdp_stats_file_path.empty()) {
      return true;
    }
    std::ifstream in(g_adaptive_fdp_stats_file_path.c_str());
    if (!in.good()) {
      return false;
    }
    std::string line1, line2;
    std::getline(in, line1);
    std::getline(in, line2);
    auto digits_only = [](std::string s) {
      std::string t;
      for (char c : s) {
        if (c >= '0' && c <= '9') {
          t.push_back(c);
        }
      }
      return t;
    };
    line1 = digits_only(std::move(line1));
    line2 = digits_only(std::move(line2));
    if (line1.empty() || line2.empty()) {
      return false;
    }
    *hbmw = std::strtoull(line1.c_str(), nullptr, 10);
    *mbmw = std::strtoull(line2.c_str(), nullptr, 10);
    return true;
  }

#if defined(__linux__)
#pragma pack(push, 1)
  struct NvmeFdpStatsLogPage {
    uint64_t hbmw[2];
    uint64_t mbmw[2];
    uint64_t mbe[2];
    uint8_t rsvd[16];
  };
#pragma pack(pop)

  bool ReadFdpHbmwMbmwFromNvme(const std::string& dev_path, uint64_t* h0,
                               uint64_t* h1, uint64_t* m0, uint64_t* m1) {
    *h0 = *h1 = *m0 = *m1 = 0;
    if (dev_path.empty()) {
      return false;
    }
    int fd = open(dev_path.c_str(), O_RDONLY);
    if (fd < 0) {
      return false;
    }
    NvmeFdpStatsLogPage log{};
    const unsigned data_len = static_cast<unsigned>(sizeof(log));
    const uint32_t numd = data_len / 4 - 1;

    struct nvme_passthru_cmd cmd = {};
    cmd.opcode = 0x02;  // Get Log Page
    cmd.nsid = 0;
    cmd.addr = reinterpret_cast<uint64_t>(&log);
    cmd.data_len = data_len;
    cmd.cdw10 = static_cast<uint32_t>(0x22u | (numd << 16));  // LID FDP Statistics
    cmd.cdw11 = 0;
    cmd.cdw12 = 0;
    cmd.cdw13 = 0;

    int err = ioctl(fd, NVME_IOCTL_ADMIN_CMD, &cmd);
    const int io_errno = errno;
    close(fd);
    if (err != 0) {
      errno = io_errno;
      return false;
    }
    *h0 = le64toh(log.hbmw[0]);
    *h1 = le64toh(log.hbmw[1]);
    *m0 = le64toh(log.mbmw[0]);
    *m1 = le64toh(log.mbmw[1]);
    return true;
  }

  bool ResolveNvmeNsDeviceForDbPath(const std::string& db_path,
                                    std::string* out_dev) {
    out_dev->clear();
    if (db_path.empty()) {
      return false;
    }
    char canon[PATH_MAX];
    if (realpath(db_path.c_str(), canon) == nullptr) {
      return false;
    }
    struct stat st;
    if (stat(canon, &st) != 0) {
      return false;
    }
    const unsigned maj = static_cast<unsigned>(major(st.st_dev));
    const unsigned min = static_cast<unsigned>(minor(st.st_dev));
    char linkpath[128];
    snprintf(linkpath, sizeof(linkpath), "/sys/dev/block/%u:%u", maj, min);
    char target[512];
    const ssize_t n = readlink(linkpath, target, sizeof(target) - 1);
    if (n <= 0) {
      return false;
    }
    target[n] = '\0';
    std::string t(target, static_cast<size_t>(n));
    const size_t slash = t.find_last_of('/');
    if (slash == std::string::npos) {
      return false;
    }
    std::string leaf = t.substr(slash + 1);
    if (leaf.size() < 4 || leaf.compare(0, 4, "nvme") != 0) {
      return false;
    }
    const size_t ppos = leaf.rfind('p');
    if (ppos != std::string::npos && ppos + 1 < leaf.size()) {
      bool all_digits = true;
      for (size_t k = ppos + 1; k < leaf.size(); ++k) {
        if (!std::isdigit(static_cast<unsigned char>(leaf[k]))) {
          all_digits = false;
          break;
        }
      }
      if (all_digits) {
        leaf = leaf.substr(0, ppos);
      }
    }
    *out_dev = "/dev/" + leaf;
    return !out_dev->empty();
  }

  void AdaptiveTryResolveFdpNvmeFromDbPathEnv(const char* db_path) {
    const char* auto_e =
        std::getenv("ROCKSDB_TOO_FAR_BUDGET_ADAPTIVE_FDP_FROM_DB_PATH");
    if (auto_e == nullptr || auto_e[0] != '1' || auto_e[1] != '\0') {
      return;
    }
    if (db_path == nullptr || db_path[0] == '\0') {
      return;
    }
    if (!g_adaptive_fdp_nvme_device.empty()) {
      return;
    }
    if (!g_adaptive_fdp_stats_file_path.empty()) {
      return;
    }
    std::string dev;
    if (ResolveNvmeNsDeviceForDbPath(db_path, &dev)) {
      g_adaptive_fdp_nvme_device = std::move(dev);
      fprintf(stderr,
              "[TwoPhaseWriteManager] ADAPTIVE FDP: resolved NVMe device %s "
              "from db_path=%s (ROCKSDB_TOO_FAR_BUDGET_ADAPTIVE_FDP_FROM_DB_PATH=1)\n",
              g_adaptive_fdp_nvme_device.c_str(), db_path);
    } else {
      fprintf(stderr,
              "[TwoPhaseWriteManager][WARN] ADAPTIVE FDP: "
              "FDP_FROM_DB_PATH=1 but could not resolve NVMe ns from db_path=%s\n",
              db_path);
    }
  }
#else
  void AdaptiveTryResolveFdpNvmeFromDbPathEnv(const char* /*db_path*/) {}
#endif

  bool ReadFdpHbmwMbmwPair(uint64_t* h0, uint64_t* h1, uint64_t* m0,
                            uint64_t* m1) {
    *h0 = *h1 = *m0 = *m1 = 0;
#if defined(__linux__)
    if (!g_adaptive_fdp_nvme_device.empty()) {
      errno = 0;
      if (ReadFdpHbmwMbmwFromNvme(g_adaptive_fdp_nvme_device, h0, h1, m0, m1)) {
        return true;
      }
      const int io_err = errno;
      static std::atomic<int> nvme_warn_left{4};
      int left = nvme_warn_left.fetch_sub(1, std::memory_order_relaxed);
      if (left > 0) {
        fprintf(stderr,
                "[TwoPhaseWriteManager][WARN] ADAPTIVE FDP: ioctl read failed "
                "for device=%s (need read access to NVMe char dev; errno=%d). "
                "Falling back to file if FDP_STATS_FILE set.\n",
                g_adaptive_fdp_nvme_device.c_str(), io_err);
      }
    }
#endif
    if (!g_adaptive_fdp_stats_file_path.empty()) {
      uint64_t hf = 0, mf = 0;
      if (!ReadFdpHbmwMbmwFromFile(&hf, &mf)) {
        return false;
      }
      *h0 = hf;
      *h1 = 0;
      *m0 = mf;
      *m1 = 0;
      return true;
    }
    return true;
  }

  void AdaptiveSnapshotWaSegmentStart() {
    g_adaptive_seg_user_b = 0;
    g_adaptive_seg_flush_b = 0;
    g_adaptive_seg_compact_b = 0;
    g_adaptive_seg_fdp_h0 = 0;
    g_adaptive_seg_fdp_h1 = 0;
    g_adaptive_seg_fdp_m0 = 0;
    g_adaptive_seg_fdp_m1 = 0;
    Statistics* st =
        g_adaptive_probe_byte_statistics.load(std::memory_order_relaxed);
    if (st != nullptr) {
      g_adaptive_seg_user_b = st->getTickerCount(BYTES_WRITTEN);
      g_adaptive_seg_flush_b = st->getTickerCount(FLUSH_WRITE_BYTES);
      g_adaptive_seg_compact_b = st->getTickerCount(COMPACT_WRITE_BYTES);
    }
    if (FdpWaSourceEnabled()) {
      (void)ReadFdpHbmwMbmwPair(&g_adaptive_seg_fdp_h0, &g_adaptive_seg_fdp_h1,
                                &g_adaptive_seg_fdp_m0, &g_adaptive_seg_fdp_m1);
    }
  }

  void AdaptiveComputeWaSlotScore(double* score_out, double* app_wa_out,
                                   double* dev_wa_out) {
    constexpr double kHuge = 1e12;
    *score_out = kHuge;
    *app_wa_out = kHuge;
    *dev_wa_out = 1.0;
    Statistics* st =
        g_adaptive_probe_byte_statistics.load(std::memory_order_relaxed);
    if (st == nullptr) {
      return;
    }
    const uint64_t u1 = st->getTickerCount(BYTES_WRITTEN);
    const uint64_t f1 = st->getTickerCount(FLUSH_WRITE_BYTES);
    const uint64_t c1 = st->getTickerCount(COMPACT_WRITE_BYTES);
    const uint64_t du =
        u1 >= g_adaptive_seg_user_b ? u1 - g_adaptive_seg_user_b : 0;
    const uint64_t df =
        f1 >= g_adaptive_seg_flush_b ? f1 - g_adaptive_seg_flush_b : 0;
    const uint64_t dc =
        c1 >= g_adaptive_seg_compact_b ? c1 - g_adaptive_seg_compact_b : 0;
    double app_wa = kHuge;
    if (du > 0) {
      app_wa = static_cast<double>(df + dc) / static_cast<double>(du);
    }
    double dev_wa = 1.0;
    if (FdpWaSourceEnabled()) {
      uint64_t h0 = 0, h1 = 0, m0 = 0, m1 = 0;
      if (ReadFdpHbmwMbmwPair(&h0, &h1, &m0, &m1)) {
        uint64_t dh0 = 0, dh1 = 0, dm0 = 0, dm1 = 0;
        FdpU128Sub(h0, h1, g_adaptive_seg_fdp_h0, g_adaptive_seg_fdp_h1, &dh0,
                   &dh1);
        FdpU128Sub(m0, m1, g_adaptive_seg_fdp_m0, g_adaptive_seg_fdp_m1, &dm0,
                   &dm1);
#if defined(__linux__) && defined(__SIZEOF_INT128__) && defined(__GNUC__)
        const FdpU128 dh = FdpU128FromPair(dh0, dh1);
        const FdpU128 dm = FdpU128FromPair(dm0, dm1);
        if (dh > 0) {
          dev_wa = static_cast<double>(static_cast<long double>(dm) /
                                       static_cast<long double>(dh));
        }
#else
        if (dh0 > 0) {
          dev_wa = static_cast<double>(dm0) / static_cast<double>(dh0);
        }
#endif
      }
    }
    *app_wa_out = app_wa;
    *dev_wa_out = dev_wa;
    if (app_wa >= kHuge * 0.5) {
      *score_out = kHuge;
    } else {
      *score_out = app_wa * dev_wa;
    }
  }

  void LogAdaptiveProbePlanToStderrIfShrink() {
    const int psz = static_cast<int>(g_adaptive_probe_slot_plan.size());
    if (psz <= 0) {
      return;
    }
    std::string plan;
    plan.reserve(static_cast<size_t>(psz) * 8);
    for (int i = 0; i < psz; i++) {
      if (i) plan.push_back(',');
      plan.append(std::to_string(g_adaptive_probe_slot_plan[static_cast<size_t>(i)]));
    }
    if (g_adaptive_triple_0510_mode) {
      fprintf(stderr,
              "[TooFarBudgetAdaptive] triple0510 next_probe_global_slots=[%s] "
              "last_win_global=%d same_win_streak=%d\n",
              plan.c_str(), g_adaptive_last_win_global, g_adaptive_win_streak);
      return;
    }
    if (!g_adaptive_probe_shrink_enabled) {
      return;
    }
    const int n = static_cast<int>(g_adaptive_probe_slots.size());
    if (psz >= n) {
      return;
    }
    fprintf(stderr,
            "[TooFarBudgetAdaptive] probe_shrink=1 next_probe_global_slots=[%s] "
            "(last_win_global=%d same_win_streak=%d min_plan_win_streak=%d "
            "last_closed_plan_min_global=%d)\n",
            plan.c_str(), g_adaptive_last_win_global, g_adaptive_win_streak,
            g_adaptive_min_plan_win_streak, g_adaptive_last_closed_plan_min_global);
  }

  void AdaptiveBeginProbeRound(TooFarBudgetState& s,
                               std::chrono::steady_clock::time_point now) {
    g_adaptive_probe_slot_plan.clear();
    const int n = static_cast<int>(g_adaptive_probe_slots.size());

    if (g_adaptive_force_win_global >= 0 && g_adaptive_force_win_global < n) {
      g_adaptive_probe_slot_plan.push_back(g_adaptive_force_win_global);
      g_adaptive_probe_index = 0;
      g_adaptive_slot_bps.clear();
      g_adaptive_slot_app_wa.clear();
      g_adaptive_slot_dev_wa.clear();
      g_adaptive_segment_start = now;
      g_adaptive_segment_byte_start = ReadAdaptiveProbeByteCounter();
      g_adaptive_segment_compaction_start =
          g_adaptive_compaction_jobs_started_total.load(
              std::memory_order_relaxed);
      if (g_adaptive_probe_metric == AdaptiveProbeMetric::kWaProduct) {
        AdaptiveSnapshotWaSegmentStart();
      }
      const int g0 = g_adaptive_probe_slot_plan[0];
      const auto& kn0 = g_adaptive_probe_slots[static_cast<size_t>(g0)];
      s.allow = kn0.allow;
      s.window = kn0.window;
      s.attempts = 0;
      s.customs_used = 0;
      s.slot_idx = -1;
      fprintf(stderr,
              "[TooFarBudgetAdaptive] force_win next_probe_global_slots=[%d] "
              "(collapsed)\n",
              g_adaptive_force_win_global);
      return;
    }

    if (g_adaptive_triple_0510_mode && n == 3) {
      if (g_adaptive_last_win_global < 0) {
        g_adaptive_probe_slot_plan.push_back(0);
        g_adaptive_probe_slot_plan.push_back(1);
        g_adaptive_probe_slot_plan.push_back(2);
      } else {
        const int w = g_adaptive_last_win_global;
        // triple0510 收窄：仅极端档（0/10 或 10/10）胜出时缩探针；5/10 胜出必全扫。
        if (w == 0) {
          g_adaptive_probe_slot_plan.push_back(0);
          g_adaptive_probe_slot_plan.push_back(1);
        } else if (w == 1) {
          g_adaptive_probe_slot_plan.push_back(0);
          g_adaptive_probe_slot_plan.push_back(1);
          g_adaptive_probe_slot_plan.push_back(2);
        } else {
          g_adaptive_probe_slot_plan.push_back(1);
          g_adaptive_probe_slot_plan.push_back(2);
        }
      }
      g_adaptive_probe_index = 0;
      g_adaptive_slot_bps.clear();
      g_adaptive_slot_app_wa.clear();
      g_adaptive_slot_dev_wa.clear();
      g_adaptive_segment_start = now;
      g_adaptive_segment_byte_start = ReadAdaptiveProbeByteCounter();
      g_adaptive_segment_compaction_start =
          g_adaptive_compaction_jobs_started_total.load(
              std::memory_order_relaxed);
      if (g_adaptive_probe_metric == AdaptiveProbeMetric::kWaProduct) {
        AdaptiveSnapshotWaSegmentStart();
      }
      const int g0 = g_adaptive_probe_slot_plan[0];
      const auto& kn0 = g_adaptive_probe_slots[static_cast<size_t>(g0)];
      s.allow = kn0.allow;
      s.window = kn0.window;
      s.attempts = 0;
      s.customs_used = 0;
      s.slot_idx = -1;
      LogAdaptiveProbePlanToStderrIfShrink();
      return;
    }

    int start = 0;
    const int end = n - 1;
    if (g_adaptive_probe_shrink_enabled && g_adaptive_last_win_global >= 0 &&
        n > 0) {
      const int w = g_adaptive_last_win_global;
      if (g_adaptive_win_streak >= 2) {
        // 同一全局档连赢≥2：起点左移一档（如两连 9/10 后从 7/10 起扫）。
        start = std::max(0, w - 1);
      } else if (g_adaptive_min_plan_win_streak >= 2 &&
                 g_adaptive_last_closed_plan_min_global >= 0) {
        // 连续两轮胜者均为当轮 plan 最左档：再左移一档后重置计数。
        start = std::max(0, g_adaptive_last_closed_plan_min_global - 1);
        g_adaptive_min_plan_win_streak = 0;
      } else if (g_adaptive_min_plan_win_streak == 1 &&
                 g_adaptive_last_closed_plan_min_global >= 0) {
        // 仅一轮「胜者=当轮 plan 最左」：下轮起点锁在该 plan_min。
        start = g_adaptive_last_closed_plan_min_global;
      } else {
        // 用各档 allow（K）算 ceil((K_lo+K_win)/2)，取第一个 K≥阈值的档为起点。
        const int k_lo = g_adaptive_probe_slots[0].allow;
        const int k_win = g_adaptive_probe_slots[static_cast<size_t>(w)].allow;
        const int thr = static_cast<int>(std::ceil(
            (static_cast<double>(k_lo) + static_cast<double>(k_win)) / 2.0));
        start = 0;
        while (start < n &&
               g_adaptive_probe_slots[static_cast<size_t>(start)].allow < thr) {
          start++;
        }
        if (start > end) {
          start = end;
        }
      }
    }
    for (int i = start; i <= end && i < n; i++) {
      g_adaptive_probe_slot_plan.push_back(i);
    }
    if (g_adaptive_probe_slot_plan.empty()) {
      for (int i = 0; i < n; i++) {
        g_adaptive_probe_slot_plan.push_back(i);
      }
    }
    g_adaptive_probe_index = 0;
    g_adaptive_slot_bps.clear();
    g_adaptive_slot_app_wa.clear();
    g_adaptive_slot_dev_wa.clear();
    g_adaptive_segment_start = now;
    g_adaptive_segment_byte_start = ReadAdaptiveProbeByteCounter();
    g_adaptive_segment_compaction_start =
        g_adaptive_compaction_jobs_started_total.load(
            std::memory_order_relaxed);
    if (g_adaptive_probe_metric == AdaptiveProbeMetric::kWaProduct) {
      AdaptiveSnapshotWaSegmentStart();
    }
    const int g0 = g_adaptive_probe_slot_plan[0];
    const auto& kn0 = g_adaptive_probe_slots[static_cast<size_t>(g0)];
    s.allow = kn0.allow;
    s.window = kn0.window;
    s.attempts = 0;
    s.customs_used = 0;
    s.slot_idx = -1;
    LogAdaptiveProbePlanToStderrIfShrink();
  }

  bool ParsePositiveIntStrict(const char* s, int min_v, int max_v, int* out) {
    if (s == nullptr || s[0] == '\0') {
      return false;
    }
    char* tail = nullptr;
    errno = 0;
    long v = std::strtol(s, &tail, 10);
    if (tail == s || errno == ERANGE || *tail != '\0') {
      return false;
    }
    if (v < static_cast<long>(min_v) || v > static_cast<long>(max_v)) {
      return false;
    }
    *out = static_cast<int>(v);
    return true;
  }

  void InitAdaptiveForceWinGlobalFromEnv() {
    g_adaptive_force_win_global = -1;
    const int n = static_cast<int>(g_adaptive_probe_slots.size());
    if (n <= 0) {
      return;
    }
    const char* force_kn =
        std::getenv("ROCKSDB_TOO_FAR_BUDGET_ADAPTIVE_FORCE_WIN_KN");
    if (force_kn != nullptr && force_kn[0] != '\0') {
      int fk = 0, fn = 0;
      const char* end = force_kn + std::strlen(force_kn);
      if (ParseSingleKNAdaptiveAllowZero(force_kn, end, &fk, &fn)) {
        for (int i = 0; i < n; i++) {
          const auto& sl = g_adaptive_probe_slots[static_cast<size_t>(i)];
          if (sl.allow == fk && sl.window == fn) {
            g_adaptive_force_win_global = i;
            break;
          }
        }
      }
      if (g_adaptive_force_win_global < 0) {
        fprintf(stderr,
                "[TwoPhaseWriteManager][WARN] "
                "ROCKSDB_TOO_FAR_BUDGET_ADAPTIVE_FORCE_WIN_KN=%s not found in "
                "PROBE_KN list; FORCE_WIN disabled.\n",
                force_kn);
      }
    }
    if (g_adaptive_force_win_global < 0) {
      const char* force_g = std::getenv(
          "ROCKSDB_TOO_FAR_BUDGET_ADAPTIVE_FORCE_WIN_GLOBAL_SLOT");
      if (force_g != nullptr && force_g[0] != '\0') {
        int gi = 0;
        if (ParsePositiveIntStrict(force_g, 0, n - 1, &gi)) {
          g_adaptive_force_win_global = gi;
        } else {
          fprintf(stderr,
                  "[TwoPhaseWriteManager][WARN] "
                  "ROCKSDB_TOO_FAR_BUDGET_ADAPTIVE_FORCE_WIN_GLOBAL_SLOT "
                  "invalid (need 0..%d); FORCE_WIN disabled. raw=%s\n",
                  n - 1, force_g);
        }
      }
    }
    if (g_adaptive_force_win_global >= 0) {
      const auto& w =
          g_adaptive_probe_slots[static_cast<size_t>(g_adaptive_force_win_global)];
      fprintf(stderr,
              "[TwoPhaseWriteManager] ADAPTIVE FORCE_WIN enabled: "
              "global_slot=%d K/N=%d/%d (fixed winner every round; probe plan "
              "collapsed to this slot only)\n",
              g_adaptive_force_win_global, w.allow, w.window);
    }
  }

  void LogAdaptiveWinnerToInfoAndStderr(
      int winning_index, int win_k, int win_n, double win_primary,
      int exploit_duration_sec, const std::vector<double>& all_scores,
      bool wa_metric, double win_app_wa, double win_dev_wa, bool forced_win) {
    std::string scores;
    scores.reserve(all_scores.size() * 24);
    for (size_t i = 0; i < all_scores.size(); i++) {
      if (i) scores.push_back(',');
      char buf[64];
      snprintf(buf, sizeof(buf), "%.6g", all_scores[i]);
      scores.append(buf);
    }
    if (wa_metric) {
      fprintf(stderr,
              "[TooFarBudgetAdaptive] round=%d winning_global_slot=%d K/N=%d/%d "
              "win_score=%.6g win_app_wa=%.6g win_dev_wa=%.6g exploit_sec=%d "
              "probe_slot_sec=%d forced_win=%d scores_wa_product=[%s]\n",
              g_adaptive_round_seq, winning_index, win_k, win_n, win_primary,
              win_app_wa, win_dev_wa, exploit_duration_sec,
              g_adaptive_probe_slot_sec, forced_win ? 1 : 0, scores.c_str());
    } else {
      fprintf(stderr,
              "[TooFarBudgetAdaptive] round=%d winning_global_slot=%d K/N=%d/%d "
              "win_bytes_per_sec=%.3f exploit_sec=%d probe_slot_sec=%d "
              "forced_win=%d scores_bps=[%s]\n",
              g_adaptive_round_seq, winning_index, win_k, win_n, win_primary,
              exploit_duration_sec, g_adaptive_probe_slot_sec,
              forced_win ? 1 : 0, scores.c_str());
    }
    Logger* lg = g_two_phase_write_manager ? g_two_phase_write_manager->GetInfoLog()
                                           : nullptr;
    if (lg != nullptr) {
      if (wa_metric) {
        ROCKS_LOG_INFO(
            lg,
            "[TooFarBudgetAdaptive] round=%d winning_global_slot=%d K/N=%d/%d "
            "win_score=%.6g win_app_wa=%.6g win_dev_wa=%.6g exploit_sec=%d "
            "probe_slot_sec=%d scores_wa_product=[%s]",
            g_adaptive_round_seq, winning_index, win_k, win_n, win_primary,
            win_app_wa, win_dev_wa, exploit_duration_sec,
            g_adaptive_probe_slot_sec, scores.c_str());
      } else {
        ROCKS_LOG_INFO(
            lg,
            "[TooFarBudgetAdaptive] round=%d winning_global_slot=%d K/N=%d/%d "
            "win_bytes_per_sec=%.3f exploit_sec=%d probe_slot_sec=%d "
            "scores_bps=[%s]",
            g_adaptive_round_seq, winning_index, win_k, win_n, win_primary,
            exploit_duration_sec, g_adaptive_probe_slot_sec, scores.c_str());
      }
    }
  }

  void ApplyAdaptiveBudgetIfNeeded(TooFarBudgetState& s) {
    if (!g_adaptive_enabled.load(std::memory_order_relaxed)) {
      return;
    }
    const int nprobe = static_cast<int>(g_adaptive_probe_slots.size());
    if (nprobe <= 0) {
      return;
    }

    auto now = std::chrono::steady_clock::now();

    if (g_adaptive_phase == AdaptiveBudgetPhase::kExploit) {
      if (now >= g_adaptive_exploit_until) {
        g_adaptive_phase = AdaptiveBudgetPhase::kProbe;
        AdaptiveBeginProbeRound(s, now);
      }
      return;
    }

    // Probe phase：槽边界由墙钟（默认）或 compaction 次数决定。
    const int psz =
        static_cast<int>(g_adaptive_probe_slot_plan.size());
    if (psz <= 0) {
      return;
    }
    double elapsed =
        std::chrono::duration<double>(now - g_adaptive_segment_start).count();
    const double need =
        static_cast<double>(g_adaptive_probe_slot_sec > 0 ? g_adaptive_probe_slot_sec
                                                          : 1);
    bool slot_boundary = false;
    if (g_adaptive_probe_advance_mode == AdaptiveProbeAdvanceMode::kWallClock) {
      slot_boundary = (elapsed >= need);
    } else {
      const uint64_t cur = g_adaptive_compaction_jobs_started_total.load(
          std::memory_order_relaxed);
      const uint64_t d = cur - g_adaptive_segment_compaction_start;
      const int needc =
          g_adaptive_probe_slot_compactions > 0 ? g_adaptive_probe_slot_compactions
                                                : 1;
      slot_boundary = (d >= static_cast<uint64_t>(needc));
    }
    if (!slot_boundary) {
      return;
    }

    const bool wa_metric =
        (g_adaptive_probe_metric == AdaptiveProbeMetric::kWaProduct);
    if (wa_metric) {
      double score = 0, app_w = 0, dev_w = 0;
      AdaptiveComputeWaSlotScore(&score, &app_w, &dev_w);
      g_adaptive_slot_bps.push_back(score);
      g_adaptive_slot_app_wa.push_back(app_w);
      g_adaptive_slot_dev_wa.push_back(dev_w);
    } else {
      // BYTES_WRITTEN / 原子路径均为累计值：段末减段起点得 delta；bps = delta / 段时长。
      const double effective_need =
          (g_adaptive_probe_advance_mode == AdaptiveProbeAdvanceMode::kWallClock)
              ? need
              : std::max(elapsed, 1e-6);
      const uint64_t end_b = ReadAdaptiveProbeByteCounter();
      const uint64_t delta =
          end_b >= g_adaptive_segment_byte_start
              ? (end_b - g_adaptive_segment_byte_start)
              : 0;
      double bps =
          (effective_need > 0) ? (static_cast<double>(delta) / effective_need) : 0.0;
      if (g_adaptive_min_bytes > 0 && delta < g_adaptive_min_bytes) {
        bps = 0.0;
      }
      g_adaptive_slot_bps.push_back(bps);
    }

    if (g_adaptive_probe_index >= psz - 1) {
      int win_loc = 0;
      bool forced_win = false;
      int win_global = 0;
      if (g_adaptive_force_win_global >= 0 &&
          g_adaptive_force_win_global < nprobe) {
        forced_win = true;
        win_global = g_adaptive_force_win_global;
        win_loc = 0;
        for (int i = 0; i < psz; i++) {
          if (g_adaptive_probe_slot_plan[static_cast<size_t>(i)] == win_global) {
            win_loc = i;
            break;
          }
        }
      } else {
        for (int i = 1; i < psz; i++) {
          if (wa_metric) {
            if (g_adaptive_slot_bps[static_cast<size_t>(i)] <
                g_adaptive_slot_bps[static_cast<size_t>(win_loc)] - 1e-12) {
              win_loc = i;
            }
          } else {
            if (g_adaptive_slot_bps[static_cast<size_t>(i)] >
                g_adaptive_slot_bps[static_cast<size_t>(win_loc)] + 1e-12) {
              win_loc = i;
            }
          }
        }
        win_global = g_adaptive_probe_slot_plan[static_cast<size_t>(win_loc)];
      }
      if (win_global == g_adaptive_prev_win_for_streak) {
        g_adaptive_win_streak++;
      } else {
        g_adaptive_win_streak = 1;
      }
      g_adaptive_prev_win_for_streak = win_global;
      g_adaptive_last_win_global = win_global;

      const int plan_min_global = g_adaptive_probe_slot_plan[0];
      if (win_global == plan_min_global) {
        g_adaptive_min_plan_win_streak++;
      } else {
        g_adaptive_min_plan_win_streak = 0;
      }
      g_adaptive_last_closed_plan_min_global = plan_min_global;

      g_adaptive_prev_round_plan = g_adaptive_probe_slot_plan;

      const auto& wkn =
          g_adaptive_probe_slots[static_cast<size_t>(win_global)];
      s.allow = wkn.allow;
      s.window = wkn.window;
      s.attempts = 0;
      s.customs_used = 0;
      s.slot_idx = -1;
      g_adaptive_round_seq++;
      int exploit_dur =
          g_adaptive_exploit_sec > 0 ? g_adaptive_exploit_sec : 1;
      if (g_adaptive_triple_0510_mode) {
        int mult = 1;
        if (g_adaptive_win_streak >= 3) {
          mult = 4;
        } else if (g_adaptive_win_streak >= 2) {
          mult = 2;
        }
        exploit_dur *= mult;
      }
      double win_app_wa = 0, win_dev_wa = 0;
      if (wa_metric &&
          static_cast<size_t>(win_loc) < g_adaptive_slot_app_wa.size()) {
        win_app_wa = g_adaptive_slot_app_wa[static_cast<size_t>(win_loc)];
        win_dev_wa = g_adaptive_slot_dev_wa[static_cast<size_t>(win_loc)];
      }
      LogAdaptiveWinnerToInfoAndStderr(
          win_global, wkn.allow, wkn.window,
          g_adaptive_slot_bps[static_cast<size_t>(win_loc)], exploit_dur,
          g_adaptive_slot_bps, wa_metric, win_app_wa, win_dev_wa, forced_win);
      g_adaptive_phase = AdaptiveBudgetPhase::kExploit;
      g_adaptive_exploit_until = now + std::chrono::seconds(exploit_dur);
      return;
    }

    g_adaptive_probe_index++;
    const int gi =
        g_adaptive_probe_slot_plan[static_cast<size_t>(g_adaptive_probe_index)];
    const auto& kn = g_adaptive_probe_slots[static_cast<size_t>(gi)];
    s.allow = kn.allow;
    s.window = kn.window;
    s.attempts = 0;
    s.customs_used = 0;
    s.slot_idx = -1;
    g_adaptive_segment_start = now;
    g_adaptive_segment_byte_start = ReadAdaptiveProbeByteCounter();
    g_adaptive_segment_compaction_start =
        g_adaptive_compaction_jobs_started_total.load(std::memory_order_relaxed);
    if (g_adaptive_probe_metric == AdaptiveProbeMetric::kWaProduct) {
      AdaptiveSnapshotWaSegmentStart();
    }
  }

  void ApplyScheduleSlotIfChanged(TooFarBudgetState& s) {
    if (!g_budget_schedule_enabled) {
      return;
    }
    int nslots = static_cast<int>(g_budget_schedule.size());
    if (nslots <= 0) {
      return;
    }
    auto now = std::chrono::steady_clock::now();
    double elapsed =
        std::chrono::duration<double>(now - g_budget_schedule_start).count();
    if (elapsed < 0) {
      elapsed = 0;
    }
    int cur = static_cast<int>(
        std::floor(elapsed / g_budget_schedule_interval_sec)) %
        nslots;
    if (cur != s.slot_idx) {
      const auto& kn = g_budget_schedule[static_cast<size_t>(cur)];
      s.allow = kn.allow;
      s.window = kn.window;
      s.attempts = 0;
      s.customs_used = 0;
      s.slot_idx = cur;
    }
  }

  void EnsureLevelBudgetsFromEnv(Statistics* db_statistics_for_adaptive,
                                 const char* adaptive_fdp_db_path) {
    std::call_once(g_level_budgets_once,
                   [](Statistics* db_stats_for_adaptive,
                      const char* adaptive_fdp_db_path_inner) {
      g_budget_schedule_enabled = false;
      g_budget_schedule.clear();
      g_budget_schedule_env_raw.assign("(unset)");
      g_too_far_budget.allow = 0;
      g_too_far_budget.window = 0;
      g_too_far_budget.attempts = 0;
      g_too_far_budget.customs_used = 0;
      g_too_far_budget.slot_idx = -1;
      g_level_budgets_from_env = false;
      g_level_budgets_env_raw.assign("(unset)");

      const char* v = std::getenv("ROCKSDB_TOO_FAR_BUDGET");
      if (v != nullptr && v[0] != '\0') {
        int allow[5];
        int window[5];
        std::string raw_tmp;
        if (ParseLevelBudgetsEnv(v, allow, window, &raw_tmp)) {
          bool multi_diff = false;
          for (int i = 1; i < 5; i++) {
            if (allow[i] != allow[0] || window[i] != window[0]) {
              multi_diff = true;
              break;
            }
          }
          if (multi_diff) {
            fprintf(stderr,
                    "[TwoPhaseWriteManager][WARN] "
                    "ROCKSDB_TOO_FAR_BUDGET has five per-level tuples; "
                    "too_far cap is now global and uses L1 tuple %d/%d only. "
                    "Prefer a single broadcast K/N (e.g. 9/10). raw=%s\n",
                    allow[0], window[0], raw_tmp.c_str());
          }
          g_too_far_budget.allow = allow[0];
          g_too_far_budget.window = window[0];
          g_level_budgets_from_env = true;
          g_level_budgets_env_raw = std::move(raw_tmp);
        } else {
          fprintf(stderr,
                  "[TwoPhaseWriteManager][WARN] "
                  "ROCKSDB_TOO_FAR_BUDGET invalid (use N or K/N, or five "
                  "comma-separated values; 1<=K<=N<=10000); cap disabled. "
                  "raw=%s\n",
                  v);
          g_level_budgets_env_raw.assign(v);
        }
      }

      const char* ada = std::getenv("ROCKSDB_TOO_FAR_BUDGET_ADAPTIVE");
      const bool want_adaptive =
          (ada != nullptr && ada[0] == '1' && ada[1] == '\0');
      g_adaptive_enabled.store(false, std::memory_order_relaxed);
      g_adaptive_probe_shrink_enabled = false;
      g_adaptive_triple_0510_mode = false;
      g_adaptive_force_win_global = -1;
      g_adaptive_prev_round_plan.clear();
      g_adaptive_probe_slot_plan.clear();
      g_adaptive_min_plan_win_streak = 0;
      g_adaptive_last_closed_plan_min_global = -1;
      g_adaptive_probe_byte_statistics.store(nullptr, std::memory_order_relaxed);
      g_adaptive_probe_slots.clear();
      g_adaptive_probe_slot_sec = 20;
      g_adaptive_exploit_sec = 180;
      g_adaptive_min_bytes = 0;
      g_adaptive_probe_kn_raw.assign("(default)");
      g_adaptive_env_note.assign("(off)");
      g_adaptive_compaction_counting_for_probe.store(false, std::memory_order_relaxed);
      g_adaptive_probe_advance_mode = AdaptiveProbeAdvanceMode::kWallClock;
      g_adaptive_probe_slot_compactions = 30;
      g_adaptive_probe_metric = AdaptiveProbeMetric::kBps;
      g_adaptive_fdp_stats_file_path.clear();
      g_adaptive_fdp_nvme_device.clear();

      if (want_adaptive) {
        const char* kn_probe = std::getenv(
            "ROCKSDB_TOO_FAR_BUDGET_ADAPTIVE_PROBE_KN");
        const char* kn_default = "0/10,5/10,10/10";
        const char* kn_use =
            (kn_probe != nullptr && kn_probe[0] != '\0') ? kn_probe : kn_default;
        std::vector<BudgetScheduleSlot> probe_tmp;
        if (ParseAdaptiveProbeSlotsEnv(kn_use, &probe_tmp) && !probe_tmp.empty()) {
          g_adaptive_probe_slots = std::move(probe_tmp);
          g_adaptive_probe_kn_raw.assign(kn_use);
          g_adaptive_triple_0510_mode =
              IsTriple0510ProbeSlots(g_adaptive_probe_slots);
          g_adaptive_prev_round_plan.clear();
          InitAdaptiveForceWinGlobalFromEnv();

          const char* slot_e =
              std::getenv("ROCKSDB_TOO_FAR_BUDGET_ADAPTIVE_PROBE_SLOT_SEC");
          int slot_sec = 20;
          if (slot_e != nullptr && slot_e[0] != '\0') {
            if (!ParsePositiveIntStrict(slot_e, 1, 86400, &slot_sec)) {
              fprintf(stderr,
                      "[TwoPhaseWriteManager][WARN] "
                      "ROCKSDB_TOO_FAR_BUDGET_ADAPTIVE_PROBE_SLOT_SEC invalid; "
                      "using 20. raw=%s\n",
                      slot_e);
              slot_sec = 20;
            }
          }
          g_adaptive_probe_slot_sec = slot_sec;

          const char* ex_e =
              std::getenv("ROCKSDB_TOO_FAR_BUDGET_ADAPTIVE_EXPLOIT_SEC");
          int ex_sec = 180;
          if (ex_e != nullptr && ex_e[0] != '\0') {
            if (!ParsePositiveIntStrict(ex_e, 1, 864000, &ex_sec)) {
              fprintf(stderr,
                      "[TwoPhaseWriteManager][WARN] "
                      "ROCKSDB_TOO_FAR_BUDGET_ADAPTIVE_EXPLOIT_SEC invalid; "
                      "using 180. raw=%s\n",
                      ex_e);
              ex_sec = 180;
            }
          }
          g_adaptive_exploit_sec = ex_sec;

          const char* min_b =
              std::getenv("ROCKSDB_TOO_FAR_BUDGET_ADAPTIVE_MIN_BYTES");
          if (min_b != nullptr && min_b[0] != '\0') {
            errno = 0;
            char* tail = nullptr;
            unsigned long long mb = std::strtoull(min_b, &tail, 10);
            if (tail != min_b && errno != ERANGE && *tail == '\0') {
              g_adaptive_min_bytes = static_cast<uint64_t>(mb);
            } else {
              fprintf(stderr,
                      "[TwoPhaseWriteManager][WARN] "
                      "ROCKSDB_TOO_FAR_BUDGET_ADAPTIVE_MIN_BYTES invalid; "
                      "ignoring. raw=%s\n",
                      min_b);
            }
          }

          const char* shrink_e =
              std::getenv("ROCKSDB_TOO_FAR_BUDGET_ADAPTIVE_PROBE_SHRINK");
          g_adaptive_probe_shrink_enabled =
              (shrink_e != nullptr && shrink_e[0] == '1' && shrink_e[1] == '\0');

          auto EnvEqIgnoreCase = [](const char* s, const char* lit) {
            if (s == nullptr) {
              return false;
            }
            while (*lit != '\0') {
              if (static_cast<unsigned char>(std::tolower(
                      static_cast<unsigned char>(*s))) !=
                  static_cast<unsigned char>(
                      std::tolower(static_cast<unsigned char>(*lit)))) {
                return false;
              }
              s++;
              lit++;
            }
            return *s == '\0';
          };

          g_adaptive_probe_advance_mode = AdaptiveProbeAdvanceMode::kWallClock;
          const char* adv_m =
              std::getenv("ROCKSDB_TOO_FAR_BUDGET_ADAPTIVE_PROBE_ADVANCE");
          if (adv_m != nullptr && adv_m[0] != '\0') {
            if (EnvEqIgnoreCase(adv_m, "compactions")) {
              g_adaptive_probe_advance_mode =
                  AdaptiveProbeAdvanceMode::kCompactions;
            }
          }

          g_adaptive_probe_slot_compactions = 30;
          const char* comp_n = std::getenv(
              "ROCKSDB_TOO_FAR_BUDGET_ADAPTIVE_PROBE_SLOT_COMPACTIONS");
          if (comp_n != nullptr && comp_n[0] != '\0') {
            int cn = 30;
            if (!ParsePositiveIntStrict(comp_n, 1, 1000000, &cn)) {
              fprintf(stderr,
                      "[TwoPhaseWriteManager][WARN] "
                      "ROCKSDB_TOO_FAR_BUDGET_ADAPTIVE_PROBE_SLOT_COMPACTIONS invalid; "
                      "using 30. raw=%s\n",
                      comp_n);
            } else {
              g_adaptive_probe_slot_compactions = cn;
            }
          }

          g_adaptive_probe_metric = AdaptiveProbeMetric::kBps;
          const char* met = std::getenv(
              "ROCKSDB_TOO_FAR_BUDGET_ADAPTIVE_PROBE_METRIC");
          if (met != nullptr && met[0] != '\0') {
            if (EnvEqIgnoreCase(met, "wa") ||
                EnvEqIgnoreCase(met, "wa_product") ||
                EnvEqIgnoreCase(met, "app_device")) {
              g_adaptive_probe_metric = AdaptiveProbeMetric::kWaProduct;
            }
          }

          const char* fdp = std::getenv(
              "ROCKSDB_TOO_FAR_BUDGET_ADAPTIVE_FDP_STATS_FILE");
          if (fdp != nullptr && fdp[0] != '\0') {
            g_adaptive_fdp_stats_file_path.assign(fdp);
          }
          const char* fdp_dev =
              std::getenv("ROCKSDB_TOO_FAR_BUDGET_ADAPTIVE_FDP_NVME");
          if (fdp_dev != nullptr && fdp_dev[0] != '\0') {
            g_adaptive_fdp_nvme_device.assign(fdp_dev);
          }

          if (g_adaptive_probe_metric == AdaptiveProbeMetric::kWaProduct &&
              db_stats_for_adaptive == nullptr) {
            fprintf(stderr,
                    "[TwoPhaseWriteManager][WARN] "
                    "ROCKSDB_TOO_FAR_BUDGET_ADAPTIVE_PROBE_METRIC=wa requires DB "
                    "Statistics; falling back to bps.\n");
            g_adaptive_probe_metric = AdaptiveProbeMetric::kBps;
          }

          const bool count_compactions =
              (g_adaptive_probe_advance_mode ==
               AdaptiveProbeAdvanceMode::kCompactions);
          g_adaptive_compaction_counting_for_probe.store(
              count_compactions, std::memory_order_relaxed);
          if (count_compactions) {
            g_adaptive_compaction_jobs_started_total.store(
                0, std::memory_order_relaxed);
          }

          AdaptiveTryResolveFdpNvmeFromDbPathEnv(adaptive_fdp_db_path_inner);

          g_adaptive_probe_byte_statistics.store(db_stats_for_adaptive,
                                                 std::memory_order_relaxed);
          g_adaptive_enabled.store(true, std::memory_order_relaxed);
          g_adaptive_env_note.assign("ADAPTIVE=1");
          g_adaptive_phase = AdaptiveBudgetPhase::kProbe;
          g_adaptive_slot_bps.clear();
          g_adaptive_slot_app_wa.clear();
          g_adaptive_slot_dev_wa.clear();
          g_adaptive_round_seq = 0;
          g_adaptive_total_user_bytes.store(0, std::memory_order_relaxed);
          g_adaptive_last_win_global = -1;
          g_adaptive_win_streak = 0;
          g_adaptive_prev_win_for_streak = -1;
          g_adaptive_min_plan_win_streak = 0;
          g_adaptive_last_closed_plan_min_global = -1;
          auto t0 = std::chrono::steady_clock::now();
          AdaptiveBeginProbeRound(g_too_far_budget, t0);
          const char* adv_str =
              (g_adaptive_probe_advance_mode == AdaptiveProbeAdvanceMode::kCompactions)
                  ? "compactions"
                  : "wall";
          const char* met_str =
              (g_adaptive_probe_metric == AdaptiveProbeMetric::kWaProduct) ? "wa"
                                                                            : "bps";
          fprintf(stderr,
                  "[TwoPhaseWriteManager] TooFarBudget ADAPTIVE enabled: "
                  "%zu probe K/N slots, probe_advance=%s probe_slot_sec=%d "
                  "probe_slot_compactions=%d probe_metric=%s fdp_nvme=%s "
                  "fdp_stats_file=%s exploit_sec=%d min_bytes=%llu PROBE_KN_raw=%s "
                  "shrink=%d triple0510=%d\n",
                  g_adaptive_probe_slots.size(), adv_str, g_adaptive_probe_slot_sec,
                  g_adaptive_probe_slot_compactions, met_str,
                  g_adaptive_fdp_nvme_device.empty()
                      ? "(none)"
                      : g_adaptive_fdp_nvme_device.c_str(),
                  g_adaptive_fdp_stats_file_path.empty()
                      ? "(none)"
                      : g_adaptive_fdp_stats_file_path.c_str(),
                  g_adaptive_exploit_sec,
                  static_cast<unsigned long long>(g_adaptive_min_bytes),
                  g_adaptive_probe_kn_raw.c_str(),
                  g_adaptive_probe_shrink_enabled ? 1 : 0,
                  g_adaptive_triple_0510_mode ? 1 : 0);
          if (g_adaptive_probe_metric == AdaptiveProbeMetric::kWaProduct) {
            fprintf(stderr,
                    "[TwoPhaseWriteManager] ADAPTIVE probe metric=wa: per-slot score "
                    "= App_WA * Device_WA (min wins); App from Statistics "
                    "(flush+compact over USER bytes); Device from delta MBMW / delta "
                    "HBMW via NVMe FDP stats log (ROCKSDB_TOO_FAR_BUDGET_ADAPTIVE_FDP_NVME "
                    "or FDP_FROM_DB_PATH=1) or two-line FDP_STATS_FILE; else Device_WA=1.\n");
          } else if (db_stats_for_adaptive != nullptr) {
            fprintf(stderr,
                    "[TwoPhaseWriteManager] ADAPTIVE probe throughput source: "
                    "Statistics::BYTES_WRITTEN (cumulative ticker; per-slot bps = "
                    "segment delta / effective slot duration)\n");
          } else {
            fprintf(stderr,
                    "[TwoPhaseWriteManager] ADAPTIVE probe throughput source: "
                    "RecordUserWriteBytesForAdaptive (DB statistics was null)\n");
          }
        } else {
          fprintf(stderr,
                  "[TwoPhaseWriteManager][WARN] "
                  "ROCKSDB_TOO_FAR_BUDGET_ADAPTIVE=1 but "
                  "ROCKSDB_TOO_FAR_BUDGET_ADAPTIVE_PROBE_KN invalid "
                  "(comma-separated K/N, K>=0 for adaptive). Adaptive disabled. "
                  "raw=%s\n",
                  kn_use);
        }
      }

      const char* sch = std::getenv("ROCKSDB_TOO_FAR_BUDGET_SCHEDULE");
      if (sch != nullptr && sch[0] != '\0') {
        if (g_adaptive_enabled.load(std::memory_order_relaxed)) {
          fprintf(stderr,
                  "[TwoPhaseWriteManager] ROCKSDB_TOO_FAR_BUDGET_ADAPTIVE=1: "
                  "ignoring ROCKSDB_TOO_FAR_BUDGET_SCHEDULE (raw=%s)\n",
                  sch);
        } else {
          std::vector<BudgetScheduleSlot> tmp;
          if (ParseBudgetScheduleEnv(sch, &tmp)) {
            double iv = 12.0;
            const char* sch_iv =
                std::getenv("ROCKSDB_TOO_FAR_BUDGET_SCHEDULE_INTERVAL_SEC");
            if (sch_iv != nullptr && sch_iv[0] != '\0') {
              errno = 0;
              char* tail = nullptr;
              iv = std::strtod(sch_iv, &tail);
              if (tail == sch_iv || errno == ERANGE || iv <= 0.0) {
                fprintf(stderr,
                        "[TwoPhaseWriteManager][WARN] "
                        "ROCKSDB_TOO_FAR_BUDGET_SCHEDULE_INTERVAL_SEC invalid; "
                        "schedule disabled. raw=%s\n",
                        sch_iv);
                tmp.clear();
              }
            }
            if (!tmp.empty()) {
              g_budget_schedule = std::move(tmp);
              g_budget_schedule_interval_sec = iv;
              g_budget_schedule_start = std::chrono::steady_clock::now();
              g_budget_schedule_enabled = true;
              g_budget_schedule_env_raw.assign(sch);
              const int nslots = static_cast<int>(g_budget_schedule.size());
              const int cur0 = 0;
              {
                const auto& kn = g_budget_schedule[static_cast<size_t>(cur0)];
                g_too_far_budget.allow = kn.allow;
                g_too_far_budget.window = kn.window;
                g_too_far_budget.attempts = 0;
                g_too_far_budget.customs_used = 0;
                g_too_far_budget.slot_idx = cur0;
              }
              fprintf(stderr,
                      "[TwoPhaseWriteManager] BUDGET_SCHEDULE enabled: %zu slots, "
                      "interval=%.1fs, total cycle=%.1fs  raw=%s\n",
                      g_budget_schedule.size(),
                      g_budget_schedule_interval_sec,
                      static_cast<double>(nslots) * g_budget_schedule_interval_sec,
                      g_budget_schedule_env_raw.c_str());
            }
          } else {
            fprintf(stderr,
                    "[TwoPhaseWriteManager][WARN] "
                    "ROCKSDB_TOO_FAR_BUDGET_SCHEDULE invalid (comma-separated K/N; "
                    "1<=K<=N<=10000). schedule disabled. raw=%s\n",
                    sch);
            g_budget_schedule_env_raw.assign(sch);
          }
        }
      }

      LogLevelBudgetsToStderrOnce();
    },
                   db_statistics_for_adaptive, adaptive_fdp_db_path);
  }

  void LogBudgetConfigToInfoIfAny(Logger* info_log) {
    if (info_log == nullptr) {
      return;
    }
    EnsureLevelBudgetsFromEnv();
    ROCKS_LOG_INFO(
        info_log,
        "[TwoPhaseWriteManager] ROCKSDB_TOO_FAR_BUDGET global K/N: "
        "%d/%d source=%s raw=%s",
        g_too_far_budget.allow, g_too_far_budget.window,
        g_level_budgets_from_env ? "env" : "default",
        g_level_budgets_env_raw.c_str());
    if (g_adaptive_enabled.load(std::memory_order_relaxed)) {
      ROCKS_LOG_INFO(
          info_log,
          "[TwoPhaseWriteManager] ROCKSDB_TOO_FAR_BUDGET_ADAPTIVE: %zu probe "
          "slots probe_slot_sec=%d exploit_sec=%d min_bytes=%llu "
          "PROBE_KN=%s",
          g_adaptive_probe_slots.size(), g_adaptive_probe_slot_sec,
          g_adaptive_exploit_sec,
          static_cast<unsigned long long>(g_adaptive_min_bytes),
          g_adaptive_probe_kn_raw.c_str());
    } else if (g_budget_schedule_enabled) {
      ROCKS_LOG_INFO(
          info_log,
          "[TwoPhaseWriteManager] ROCKSDB_TOO_FAR_BUDGET_SCHEDULE: %zu slots, "
          "interval_sec=%.3f cycle_sec=%.3f raw=%s (overrides static BUDGET "
          "allow/window per slot; global counter)",
          g_budget_schedule.size(),
          g_budget_schedule_interval_sec,
          static_cast<double>(g_budget_schedule.size()) *
              g_budget_schedule_interval_sec,
          g_budget_schedule_env_raw.c_str());
    }
  }

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
  if (handle_write_policy_ == HandleWritePolicy::kNoFdp) {
    return kNoFdpUnifiedHandle;
  }
  if (handle_write_policy_ == HandleWritePolicy::kNativeBase) {
    return NativeCompactionMetadataHandle(level);
  }
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
  if (handle_write_policy_ == HandleWritePolicy::kHashBase) {
    return GetHandleByHash(file_number, level);
  }
  return LevelToHandle(level);
}

int TwoPhaseWriteManager::MapLifetimeToHandle(double predicted_lifetime_seconds,
                                              int level) {
  if (handle_write_policy_ == HandleWritePolicy::kNoFdp) {
    (void)predicted_lifetime_seconds;
    return kNoFdpUnifiedHandle;
  }
  if (handle_write_policy_ == HandleWritePolicy::kNativeBase) {
    (void)predicted_lifetime_seconds;
    return NativeCompactionMetadataHandle(level);
  }
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
  EnsureCompactionThresholdsFromEnv();
  if (info_log_) {
    static std::atomic<bool> logged_too_far_thresholds_once{false};
    if (!logged_too_far_thresholds_once.exchange(true)) {
      ROCKS_LOG_INFO(
          info_log_,
          "[TwoPhaseWriteManager] too_far thresholds (sec) h7-h11: "
          "%.3f %.3f %.3f %.3f %.3f  source=%s  raw=%s",
          g_compaction_threshold_7_to_11[0], g_compaction_threshold_7_to_11[1],
          g_compaction_threshold_7_to_11[2], g_compaction_threshold_7_to_11[3],
          g_compaction_threshold_7_to_11[4],
          g_compaction_threshold_from_env ? "env" : "default",
          g_compaction_threshold_env_raw.c_str());
    }
  }
  switch (handle) {
    case 3:  // native-base MEDIUM：与 handle 7 同档
      return g_compaction_threshold_7_to_11[0];
    case 4:  // native-base LONG：与 handle 9 同档
      return g_compaction_threshold_7_to_11[2];
    case 5:  // native-base EXTREME：仍为 +inf
      return kCompactionHandle12Threshold;
    case 6: return kCompactionHandle6Threshold;
    case 7: return g_compaction_threshold_7_to_11[0];
    case 8: return g_compaction_threshold_7_to_11[1];
    case 9: return g_compaction_threshold_7_to_11[2];
    case 10: return g_compaction_threshold_7_to_11[3];
    case 11: return g_compaction_threshold_7_to_11[4];
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
  if (!initialized_.load() || !enable_phase2_) {
    return;
  }
  const bool handle_ok =
      (target_handle >= 6 && target_handle <= 12) ||
      (handle_write_policy_ == HandleWritePolicy::kNativeBase && target_handle >= 3 &&
       target_handle <= 5);
  if (!handle_ok) {
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
  if (handle < 3 || handle > 12) return false;
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
    const bool ok_for_too_far =
        (handle >= 7 && handle <= 12) ||
        (handle_write_policy_ == HandleWritePolicy::kNativeBase && handle >= 3 &&
         handle <= 5);
    if (!ok_for_too_far) {
      if (handle >= 3 && handle <= 12) {
        ROCKS_LOG_WARN(info_log_,
            "[GetAllTooFarFiles] file #%" PRIu64 " level=%d handle=%d skipped "
            "(policy/handle mismatch for too-far scan)",
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
  // Sort by absolute excess over threshold (seconds), descending: pick the
  // file that exceeds its threshold by the largest margin first.
  std::sort(result.begin(), result.end(),
            [now](const FileLifetimeInfo& a, const FileLifetimeInfo& b) {
              auto excess = [now](const FileLifetimeInfo& x) {
                if (x.creation_time == 0) {
                  return -std::numeric_limits<double>::infinity();
                }
                double age = (now - x.creation_time) / 1000000.0;
                return age - x.threshold_seconds;
              };
              return excess(a) > excess(b);
            });
  ROCKS_LOG_INFO(info_log_,
      "[GetAllTooFarFiles] level=%d: files_on_level=%zu, with_metadata=%d, being_compacted_skipped=%d, too_far_count=%zu",
      level, files.size(), with_metadata, skipped_being_compacted, result.size());
  return result;
}

CustomReservationResult TwoPhaseWriteManager::TryReserveCustomForLevel(int level) {
  EnsureLevelBudgetsFromEnv();
  if (level < 1 || level > 5) {
    return CustomReservationResult::kDisabled;
  }
  auto& s = g_too_far_budget;
  if (s.window <= 0 || s.allow < 0) {
    return CustomReservationResult::kDisabled;
  }
  std::lock_guard<std::mutex> g(s.mu);
  if (g_adaptive_enabled.load(std::memory_order_relaxed)) {
    ApplyAdaptiveBudgetIfNeeded(s);
  } else {
    ApplyScheduleSlotIfChanged(s);
  }
  if (s.window <= 0 || s.allow < 0) {
    return CustomReservationResult::kDisabled;
  }
  if (s.allow == 0 || s.customs_used >= s.allow) {
    return CustomReservationResult::kBlocked;
  }
  s.customs_used++;
  return CustomReservationResult::kAcquired;
}

void TwoPhaseWriteManager::ReleaseCustomReservation(int level) {
  if (level < 1 || level > 5) {
    return;
  }
  auto& s = g_too_far_budget;
  if (s.window <= 0 || s.allow < 0) {
    return;
  }
  std::lock_guard<std::mutex> g(s.mu);
  if (s.customs_used > 0) {
    s.customs_used--;
  }
}

void TwoPhaseWriteManager::NoteLevelCompactionPicked(int level) {
  EnsureLevelBudgetsFromEnv();
  if (level < 1 || level > 5) {
    return;
  }
  auto& s = g_too_far_budget;
  if (s.window <= 0 || s.allow < 0) {
    return;
  }
  std::lock_guard<std::mutex> g(s.mu);
  if (g_adaptive_enabled.load(std::memory_order_relaxed)) {
    ApplyAdaptiveBudgetIfNeeded(s);
  } else {
    ApplyScheduleSlotIfChanged(s);
  }
  if (s.window <= 0 || s.allow < 0) {
    return;
  }
  s.attempts++;
  if (s.attempts >= s.window) {
    s.attempts = 0;
    s.customs_used = 0;
  }
}

void TwoPhaseWriteManager::RecordUserWriteBytesForAdaptive(
    uint64_t user_write_bytes) {
  if (!enable_phase2_ || user_write_bytes == 0) {
    return;
  }
  if (!g_adaptive_enabled.load(std::memory_order_relaxed)) {
    return;
  }
  if (g_adaptive_probe_byte_statistics.load(std::memory_order_relaxed) !=
      nullptr) {
    return;
  }
  g_adaptive_total_user_bytes.fetch_add(user_write_bytes,
                                        std::memory_order_relaxed);
}

void TwoPhaseWriteManager::GetHandleBounds(int handle, double* lower_bound, double* upper_bound) const {
  // 由 compaction 上限推导区间（仅用于兼容调用方，init 已不打印区间）
  // A 方案：无 case 6，handle 6 由 default 处理（upper_bound = kCompactionHandle6Threshold = +inf）
  if (lower_bound == nullptr || upper_bound == nullptr) {
    return;
  }
  switch (handle) {
    case 3:
      *lower_bound = 0.0;
      *upper_bound = GetHandleThreshold(7);
      break;
    case 4:
      *lower_bound = GetHandleThreshold(7);
      *upper_bound = GetHandleThreshold(9);
      break;
    case 5:
      *lower_bound = GetHandleThreshold(9);
      *upper_bound = GetHandleThreshold(12);
      break;
    case 6:
      *lower_bound = 0.0;
      *upper_bound = GetHandleThreshold(6);
      break;
    case 7:
      *lower_bound = 0.0;
      *upper_bound = GetHandleThreshold(7);
      break;
    case 8:
      *lower_bound = GetHandleThreshold(7);
      *upper_bound = GetHandleThreshold(8);
      break;
    case 9:
      *lower_bound = GetHandleThreshold(8);
      *upper_bound = GetHandleThreshold(9);
      break;
    case 10:
      *lower_bound = GetHandleThreshold(9);
      *upper_bound = GetHandleThreshold(10);
      break;
    case 11:
      *lower_bound = GetHandleThreshold(10);
      *upper_bound = GetHandleThreshold(11);
      break;
    case 12:
      *lower_bound = GetHandleThreshold(11);
      *upper_bound = GetHandleThreshold(12);
      break;
    default:
      ROCKS_LOG_WARN(info_log_, "[TwoPhaseWriteManager] GetHandleBounds: invalid handle %d, using handle 6 bounds", handle);
      *lower_bound = 0.0;
      *upper_bound = GetHandleThreshold(6);
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

  if (handle_write_policy_ == HandleWritePolicy::kNoFdp) {
    ROCKS_LOG_INFO(info_log_,
                   "[FILE_HANDLE] file #%" PRIu64 " level=%d -> handle %d "
                   "(no-fdp: all levels unified)",
                   file_number, level, kNoFdpUnifiedHandle);
    return kNoFdpUnifiedHandle;
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

