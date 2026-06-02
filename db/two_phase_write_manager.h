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
class Logger;
class Statistics;
class VersionStorageInfo;

// 用于自定义 Compaction 选文件：当前存活时间超过阈值的文件信息
struct FileLifetimeInfo {
  uint64_t file_number{0};
  int handle{0};
  uint64_t creation_time{0};  // 微秒
  double threshold_seconds{0.0};

  double GetCurrentAge() const;
  double GetExcessOverThreshold() const;
};

// 文件元数据（仅用于 compaction 时判断 too-far 及 Phase2 重写；路径由 db_path_+file_number 推导）
// L0–L6 均登记，用于 trivial move 时判断是否 too-far 才重写
struct FileMetadata {
  uint64_t creation_time;   // 创建时间（微秒）
  int target_handle;        // 目标 handle (6–12)
  int level;                // 所属 level (0–6)，-1 表示未知
};

// CustomCompactionPicker：too-far budget（L1–L5 每 N 次 compaction pick 至多 K 次 custom；
// ROCKSDB_TOO_FAR_BUDGET 支持 "N"（=1/N）或 "K/N"；自适应 PROBE_KN 允许 K=0（0/N）。
// window<=0 或 allow<0 表示未开启 cap；window>0 且 allow==0 表示本窗口 0 次 custom。）
enum class CustomReservationResult {
  kDisabled,  // 未开启 cap（budget 未设：window<=0 或 allow<0）
  kAcquired,  // 已抢占本 cycle 的 custom 名额（customs_used 已自增）
  kBlocked,   // 本 cycle 不允许 custom（K=0 或 customs_used>=K）
};

// 两阶段写入管理器
class TwoPhaseWriteManager {
 public:
  TwoPhaseWriteManager();
  ~TwoPhaseWriteManager();

  // 初始化（加载模型等）
  // enable_phase2: 是否启用Phase 2的两阶段写入（只有Phase 2才需要预测和二次写入）
  // info_log: RocksDB 日志对象，用于输出日志到 LOG 文件
  // db_statistics: 可选；非空时自适应探针用 Statistics::BYTES_WRITTEN 与写路径 RecordTick
  //   一致；为空则回退为 RecordUserWriteBytesForAdaptive 原子累加。
  Status Initialize(const std::string& model_dir, const std::string& db_path,
                   bool enable_phase2 = false, Logger* info_log = nullptr,
                   Statistics* db_statistics = nullptr);

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

  // ROCKSDB_HASH_HANDLE=3 / native / native-base：Optimized classification（粗分档，与 TorFS 对比用）
  bool IsNativeBasePolicy() const;

  // 按输出 level 一对一到 handle：L0→6，L1–L6→7–12，否则 -1。kLevelBase 下
  // GetTargetHandleForCompactionOutputMetadata 等走此映射；Compaction 在
  // ROCKSDB_ML_PREDICT=1 且预测成功时 L1–5 另见 MapLifetimeToHandle。
  static int LevelToHandle(int level);

  // 仅按 file_number + level 做 hash，将 L1–L6 均匀映射到 handle 7–12（7 + hash%6）；与预测解耦。L0 由调用方固定 6。
  int GetHandleByHash(uint64_t file_number, int level) const;

  // 获取文件的target handle（用于自定义compaction排序）
  int GetFileTargetHandle(uint64_t file_number) const;
  
  // 获取 handle 的 too-far 年龄阈值（秒）：handle 7–11 对应 L1–L5。
  // 可选环境变量 ROCKSDB_TOO_FAR_THRESHOLDS_SEC="t7,t8,t9,t10,t11"（5 个逗号分隔
  // 非负浮点，顺序对应 handle7..11，不要求单调递增）；未设或解析失败则默认 5,10,15,60,100。h6/h12 仍为 +inf。
  // 进程内解析一次；Initialize 与 stderr 会打印生效值。
  double GetHandleThreshold(int handle) const;

  // 获取该 level 上所有“超过阈值”的文件（用于 CustomCompactionPicker）
  std::vector<FileLifetimeInfo> GetAllTooFarFiles(int level,
                                                   VersionStorageInfo* vstorage);

  // L1–L5：ROCKSDB_TOO_FAR_BUDGET 周期间至多一次 kLevelTooFarFiles custom compaction。
  // 仅 level∈[1,5] 受控；其它 level 一律返回 kDisabled。
  CustomReservationResult TryReserveCustomForLevel(int level);
  // 仅在 TryReserveCustomForLevel 返回 kAcquired 后、最终未能产出 Compaction 时调用。
  void ReleaseCustomReservation(int level);
  // 每次 level∈[1,5] 上确实选出 compaction（custom 或 baseline）后调用一次。
  void NoteLevelCompactionPicked(int level);

  // 自适应 too-far budget：若 Initialize 传入 Statistics，则探针用 BYTES_WRITTEN；
  // 否则在写路径上按 WriteBatch 字节原子累加。
  void RecordUserWriteBytesForAdaptive(uint64_t user_write_bytes);

  // 供自适应日志写入 RocksDB LOG（可为 nullptr）。
  Logger* GetInfoLog() const { return info_log_; }

  // 登记文件元数据（未走预测路径时调用，便于 GetAllTooFarFiles 用 creation_time 判断 too-far）
  // 用于 L0 flush、L6 固定、L1–L5 按 level 分配等场景
  void RegisterFileMetadata(uint64_t file_number, int level, int target_handle,
                            uint64_t file_size = 0);

  // 是否对 compaction 输出做“仅注册元数据”（不写 buffer、不重写）。
  // 由环境变量 ROCKSDB_PHASE2_REGISTER_METADATA 控制：1 或未设则 true，0 则 false。
  bool ShouldRegisterCompactionOutputMetadata() const;

  // 仅按 level 登记 compaction 输出文件元数据（便于 GetAllTooFarFiles / 自定义 compaction）。不写 buffer、不重写。
  void RegisterCompactionOutputFileMetadata(uint64_t file_number, int level,
                                            uint64_t file_size);

  // Phase2 开启时由 Initialize 解析的 ROCKSDB_HASH_HANDLE 决定元数据/写 hint 用的整型值：
  //   0 = level-base（Lk→6+k），1 = hash-base（L1–6→7–12），2/no-fdp = 全 level 单 handle（6），
  //   3/native/native-base = Optimized 分档：返回值即 Env::WriteLifeTimeHint
  //   （MEDIUM=3→L0–L3，LONG=4→L4，EXTREME=5→≥L5；WAL=SHORT 仍由 CalculateWALWriteHint）。
  // level-base（0）：本函数对 L1–6 即 LevelToHandle；CompactionJob 在 ROCKSDB_ML_PREDICT=1
  // 且预测成功时，L1–5 会先 MapLifetimeToHandle（寿命分桶到 7–11），与「一层一 handle」不同。
  int GetTargetHandleForCompactionOutputMetadata(uint64_t file_number,
                                                 int level) const;

  // 是否执行 Phase2 重写（buffer 写出到目标 handle）。
  // 由环境变量 ROCKSDB_PHASE2_REWRITE 控制：0 则 false，1 或未设则 true。
  bool ShouldDoPhase2Rewrite() const;

  // 解耦封装：若应注册则注册 compaction 输出元数据，由 ROCKSDB_PHASE2_REGISTER_METADATA 控制。
  void MaybeRegisterCompactionOutputMetadata(uint64_t file_number, int level,
                                             uint64_t file_size);

  // 解耦封装：若应做 Phase2 重写则重写，否则释放 buffer；由 ROCKSDB_PHASE2_REWRITE 控制。
  void MaybeDoPhase2Rewrite(uint64_t file_number);

  // 将某文件的 trivial move 计数 +1 并返回新值；仅当返回值为偶数时才应执行重写（第 2、4、6… 次重写）
  uint64_t IncrementAndGetTrivialMoveCount(uint64_t file_number);

  // 判断某文件是否已“too far”（age > GetHandleThreshold(handle)）；无元数据或未超阈值返回 false
  bool IsFileTooFar(uint64_t file_number) const;

  // trivial move 重写专用：age > 5 * GetHandleThreshold(handle)（自定义 compaction 仍用 IsFileTooFar / GetAllTooFarFiles）
  bool IsFileTooFarForTrivialMoveRewrite(uint64_t file_number) const;

  // 在文件打开前预测 handle（新方案：无需二次写入）
  // 使用精确的 smallest（first_key）+ 估算的 largest 来计算特征和预测
  // 参数：
  //   - file_number: 文件编号
  //   - level: 输出层级
  //   - first_key: 文件的第一个 key（精确的 smallest）
  //   - sub_compact: SubcompactionState 指针（用于获取边界信息）
  //   - cfd: ColumnFamilyData 指针（用于获取 version 信息）
  //   - db_mutex: 数据库互斥锁
  // 返回：预测的 handle（通常为 6–12；native-base 下为 3–5），失败返回 -1
  int PredictHandleBeforeWrite(uint64_t file_number, int level,
                               const Slice& first_key,
                               void* sub_compact,
                               void* cfd,
                               void* db_mutex);

  // 仅收集特征并打印，不进行预测（用于调试）
  // 参数同 PredictHandleBeforeWrite
  void CollectAndPrintFeaturesBeforeWrite(uint64_t file_number, int level,
                                          const Slice& first_key,
                                          void* sub_compact,
                                          void* cfd,
                                          void* db_mutex);

  // 映射生命周期到 handle（供 compaction_job 等调用）
  int MapLifetimeToHandle(double predicted_lifetime_seconds, int level);

 private:
  // 模型加载
  Status LoadModels(const std::string& model_dir);

  // 预测生命周期
  double PredictLifetime(int level, const std::vector<double>& features);
  
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
  // Phase2 时由 ROCKSDB_HASH_HANDLE 在 Initialize 中解析：
  // "0"=level-base，"1"=hash-base，"2"/"no-fdp"=单 handle，"3"/"native"/"native-base"=Optimized 档
  enum class HandleWritePolicy : uint8_t {
    kLevelBase = 0,
    kHashBase = 1,
    kNoFdp = 2,
    kNativeBase = 3,
  };
  HandleWritePolicy handle_write_policy_{HandleWritePolicy::kLevelBase};
  Logger* info_log_;     // RocksDB 日志对象，用于输出日志到 LOG 文件
  
  // 文件元数据映射（仅用于两阶段写入，不管理删除）
  std::unordered_map<uint64_t, FileMetadata> file_metadata_;
  mutable std::shared_mutex metadata_mutex_;  // 读写锁：读操作（查找）可以并发，写操作（插入/删除/更新）需要独占
  // 每个文件累计被 trivial move 的次数（含未在 file_metadata_ 中的文件）；每 2 次才重写 1 次
  std::unordered_map<uint64_t, uint64_t> trivial_move_count_;
  
  // 内存缓冲区映射（存储文件数据）
  std::unordered_map<uint64_t, std::string> memory_buffers_;
  std::shared_mutex buffer_mutex_;  // 读写锁：保护内存缓冲区

  // 模型相关（需要根据实际模型库实现）
  // 这里使用void*作为占位符，实际需要根据使用的模型库（如LightGBM C API）定义
  void* models_[7];  // level 0-6的模型
  void* scalers_[7];  // level 0-6的scaler
  
  std::atomic<bool> initialized_;

  // 特征子集（若加载成功则只传 N 维给 Python，否则传 69 维）
  std::vector<int> feature_subset_indices_;
  size_t n_features_subset_{0};  // 0 表示使用全部 69 维

  // ML 开销统计（特征收集 + Python 预测）
  std::atomic<uint64_t> total_feature_calc_us_{0};
  std::atomic<uint64_t> total_python_predict_us_{0};
  std::atomic<uint64_t> predict_call_count_{0};

  // 从 model_dir 加载 feature_subset.txt，成功则设置 feature_subset_indices_ 与 n_features_subset_
  void LoadFeatureSubset(const std::string& model_dir);
};

// 全局单例（由DBImpl管理生命周期）
extern TwoPhaseWriteManager* g_two_phase_write_manager;

// 每个 compaction job 开始时由 CompactionJob 调用；仅在 ADAPTIVE 且
// ROCKSDB_TOO_FAR_BUDGET_ADAPTIVE_PROBE_ADVANCE=compactions 时递增计数。
void TwoPhaseAdaptiveNoteCompactionJobStarted();

}  // namespace ROCKSDB_NAMESPACE

