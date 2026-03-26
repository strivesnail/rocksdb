//  Copyright (c) 2011-present, Facebook, Inc.  All rights reserved.
//  This source code is licensed under both the GPLv2 (found in the
//  COPYING file in the root directory) and Apache 2.0 License
//  (found in the LICENSE.Apache file in the root directory).

#pragma once

#include "db/compaction/compaction_picker_level.h"

namespace ROCKSDB_NAMESPACE {

// 基于预测删除时间的自定义 Compaction 选择器
// 继承自 LevelCompactionPicker，在特定条件下使用优先队列选择文件
class CustomCompactionPicker : public LevelCompactionPicker {
 public:
  CustomCompactionPicker(const ImmutableOptions& ioptions,
                         const InternalKeyComparator* icmp)
      : LevelCompactionPicker(ioptions, icmp) {}

  Compaction* PickCompaction(
      const std::string& cf_name, const MutableCFOptions& mutable_cf_options,
      const MutableDBOptions& mutable_db_options,
      const std::vector<SequenceNumber>& existing_snapshots,
      const SnapshotChecker* snapshot_checker,
      VersionStorageInfo* vstorage, LogBuffer* log_buffer,
      const std::string& full_history_ts_low,
      bool require_max_output_level = false) override;

  Compaction* TryCustomCompactionForLevel(
      int start_level, int output_level, VersionStorageInfo* vstorage,
      const std::string& cf_name, const MutableCFOptions& mutable_cf_options,
      const MutableDBOptions& mutable_db_options, LogBuffer* log_buffer,
      const std::string& full_history_ts_low, double start_level_score,
      std::string* normal_reason = nullptr) override;

 private:
  // 从优先队列选择文件进行 Compaction
  Compaction* SelectFileFromPriorityQueue(
      const std::string& cf_name, const MutableCFOptions& mutable_cf_options,
      const MutableDBOptions& mutable_db_options,
      const std::vector<SequenceNumber>& existing_snapshots,
      const SnapshotChecker* snapshot_checker,
      VersionStorageInfo* vstorage, LogBuffer* log_buffer,
      const std::string& full_history_ts_low,
      uint64_t file_number, int level,
      CompactionReason compaction_reason, double compaction_score);
};

}  // namespace ROCKSDB_NAMESPACE









