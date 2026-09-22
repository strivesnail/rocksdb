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


constexpr size_t kCacheLineSize = 64;


struct alignas(kCacheLineSize) LevelCompactionPriState {



  std::atomic<uint64_t> compaction_count{0};


  char padding[kCacheLineSize - sizeof(std::atomic<uint64_t>)];
};

static_assert(sizeof(LevelCompactionPriState) == kCacheLineSize,
              "LevelCompactionPriState must be cache-line aligned");



class CustomCompactionPriManager {
 public:
  CustomCompactionPriManager();
  ~CustomCompactionPriManager();



  void Initialize(int num_levels, Logger* info_log = nullptr);



  static bool IsEnabled();




  bool ShouldUseCustomPri(int level) const;



  void IncrementCompactionCount(int level);


  uint64_t GetCompactionCount(int level) const;



  static int GetCompactionRatio();


  void Reset();

 private:


  std::vector<std::unique_ptr<LevelCompactionPriState>> level_states_;


  mutable std::shared_mutex rw_mutex_;


  int num_levels_;


  std::atomic<bool> initialized_{false};


  Logger* info_log_;
};


extern CustomCompactionPriManager* g_custom_compaction_pri_manager;

}  // namespace ROCKSDB_NAMESPACE

