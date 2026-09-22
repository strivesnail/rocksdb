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


struct FileLifetimeInfo {
  uint64_t file_number{0};
  int handle{0};
  uint64_t creation_time{0};
  double threshold_seconds{0.0};

  double GetCurrentAge() const;
  double GetExcessOverThreshold() const;
};



struct FileMetadata {
  uint64_t creation_time;
  int target_handle;
  int level;
};




enum class CustomReservationResult {
  kDisabled,
  kAcquired,
  kBlocked,
};


class TwoPhaseWriteManager {
 public:
  TwoPhaseWriteManager();
  ~TwoPhaseWriteManager();






  Status Initialize(const std::string& model_dir, const std::string& db_path,
                   bool enable_phase2 = false, Logger* info_log = nullptr,
                   Statistics* db_statistics = nullptr);



  bool HandleFileCreation(uint64_t file_number, int level,
                          const std::string& file_path);


  Status CreateMemoryBuffer(uint64_t file_number);


  Status AppendToMemoryBuffer(uint64_t file_number, const Slice& data);



  std::unique_ptr<FSWritableFile> CreateMemoryWritableFile(uint64_t file_number);


  Status FinishMemoryBuffer(uint64_t file_number);


  Status GetMemoryBufferData(uint64_t file_number, std::string* data);


  void ReleaseMemoryBuffer(uint64_t file_number);


  Status CollectFeaturesAndPredict(uint64_t file_number, int level,
                                   const std::vector<double>& features);


  Status RewriteFileToTargetHandle(uint64_t file_number);


  void Shutdown();


  bool IsInitialized() const { return initialized_.load(); }


  bool IsPhase2Enabled() const { return enable_phase2_; }


  bool IsNativeBasePolicy() const;




  static int LevelToHandle(int level);


  int GetHandleByHash(uint64_t file_number, int level) const;


  int GetFileTargetHandle(uint64_t file_number) const;





  double GetHandleThreshold(int handle) const;


  std::vector<FileLifetimeInfo> GetAllTooFarFiles(int level,
                                                   VersionStorageInfo* vstorage);



  CustomReservationResult TryReserveCustomForLevel(int level);

  void ReleaseCustomReservation(int level);

  void NoteLevelCompactionPicked(int level);



  void RecordUserWriteBytesForAdaptive(uint64_t user_write_bytes);


  Logger* GetInfoLog() const { return info_log_; }



  void RegisterFileMetadata(uint64_t file_number, int level, int target_handle,
                            uint64_t file_size = 0);



  bool ShouldRegisterCompactionOutputMetadata() const;


  void RegisterCompactionOutputFileMetadata(uint64_t file_number, int level,
                                            uint64_t file_size);







  int GetTargetHandleForCompactionOutputMetadata(uint64_t file_number,
                                                 int level) const;



  bool ShouldDoPhase2Rewrite() const;


  void MaybeRegisterCompactionOutputMetadata(uint64_t file_number, int level,
                                             uint64_t file_size);


  void MaybeDoPhase2Rewrite(uint64_t file_number);


  uint64_t IncrementAndGetTrivialMoveCount(uint64_t file_number);


  bool IsFileTooFar(uint64_t file_number) const;


  bool IsFileTooFarForTrivialMoveRewrite(uint64_t file_number) const;











  int PredictHandleBeforeWrite(uint64_t file_number, int level,
                               const Slice& first_key,
                               void* sub_compact,
                               void* cfd,
                               void* db_mutex);



  void CollectAndPrintFeaturesBeforeWrite(uint64_t file_number, int level,
                                          const Slice& first_key,
                                          void* sub_compact,
                                          void* cfd,
                                          void* db_mutex);


  int MapLifetimeToHandle(double predicted_lifetime_seconds, int level);

 private:

  Status LoadModels(const std::string& model_dir);


  double PredictLifetime(int level, const std::vector<double>& features);




  void GetHandleBounds(int handle, double* lower_bound, double* upper_bound) const;


  Status WriteMemoryBufferToFile(uint64_t file_number,
                                 const std::string& buffer_data,
                                 const std::string& target_path,
                                 int target_handle);


  Status MoveFileAtomically(const std::string& src_path,
                           const std::string& dst_path, int target_handle);


  std::string db_path_;
  std::string model_dir_;
  bool enable_phase2_;


  enum class HandleWritePolicy : uint8_t {
    kLevelBase = 0,
    kHashBase = 1,
    kNoFdp = 2,
    kNativeBase = 3,
  };
  HandleWritePolicy handle_write_policy_{HandleWritePolicy::kLevelBase};
  Logger* info_log_;


  std::unordered_map<uint64_t, FileMetadata> file_metadata_;
  mutable std::shared_mutex metadata_mutex_;

  std::unordered_map<uint64_t, uint64_t> trivial_move_count_;


  std::unordered_map<uint64_t, std::string> memory_buffers_;
  std::shared_mutex buffer_mutex_;



  void* models_[7];
  void* scalers_[7];

  std::atomic<bool> initialized_;


  std::vector<int> feature_subset_indices_;
  size_t n_features_subset_{0};


  std::atomic<uint64_t> total_feature_calc_us_{0};
  std::atomic<uint64_t> total_python_predict_us_{0};
  std::atomic<uint64_t> predict_call_count_{0};


  void LoadFeatureSubset(const std::string& model_dir);
};


extern TwoPhaseWriteManager* g_two_phase_write_manager;



void TwoPhaseAdaptiveNoteCompactionJobStarted();

}  // namespace ROCKSDB_NAMESPACE

