//  Copyright (c) 2011-present, Facebook, Inc.  All rights reserved.
//  This source code is licensed under both the GPLv2 (found in the
//  COPYING file in the root directory) and Apache 2.0 License
//  (found in the LICENSE.Apache file in the root directory).

#include "db/compaction/compaction_picker_custom.h"

#include <string>
#include <vector>
#include <unordered_set>
#include <cinttypes>

#include "db/compaction/compaction.h"
#include "db/custom_compaction_pri_manager.h"
#include "db/two_phase_write_manager.h"
#include "db/version_set.h"
#include "logging/log_buffer.h"
#include "logging/logging.h"
#include "options/cf_options.h"

namespace ROCKSDB_NAMESPACE {

namespace {
void NoteCompactionStartLevelIfNeeded(Compaction* c) {
  if (c == nullptr || g_two_phase_write_manager == nullptr) {
    return;
  }
  const int sl = c->start_level();
  if (sl >= 1 && sl <= 5) {
    g_two_phase_write_manager->NoteLevelCompactionPicked(sl);
  }
}
}  // namespace

Compaction* CustomCompactionPicker::PickCompaction(
    const std::string& cf_name, const MutableCFOptions& mutable_cf_options,
    const MutableDBOptions& mutable_db_options,
    const std::vector<SequenceNumber>& existing_snapshots,
    const SnapshotChecker* snapshot_checker,
    VersionStorageInfo* vstorage, LogBuffer* log_buffer,
    const std::string& full_history_ts_low,
    bool require_max_output_level) {
  const char* env_pred = getenv("ROCKSDB_ENABLE_PREDICTED_COMPACTION");
  const char* env_phase2 = getenv("ROCKSDB_ENABLE_PHASE2");
  bool use_custom_compaction =
      (env_pred != nullptr && std::string(env_pred) == "1") ||
      (env_phase2 != nullptr && std::string(env_phase2) == "1");
  if (!use_custom_compaction) {
    Compaction* c = LevelCompactionPicker::PickCompaction(
        cf_name, mutable_cf_options, mutable_db_options, existing_snapshots,
        snapshot_checker, vstorage, log_buffer, full_history_ts_low,
        require_max_output_level);
    NoteCompactionStartLevelIfNeeded(c);
    return c;
  }
  if (!g_two_phase_write_manager || !g_two_phase_write_manager->IsInitialized() ||
      !g_two_phase_write_manager->IsPhase2Enabled()) {
    Compaction* c = LevelCompactionPicker::PickCompaction(
        cf_name, mutable_cf_options, mutable_db_options, existing_snapshots,
        snapshot_checker, vstorage, log_buffer, full_history_ts_low,
        require_max_output_level);
    NoteCompactionStartLevelIfNeeded(c);
    return c;
  }
  // Delegate to parent: builder will call TryCustomCompactionForLevel when
  // level is chosen and start_level_ != 0.
  Compaction* c = LevelCompactionPicker::PickCompaction(
      cf_name, mutable_cf_options, mutable_db_options, existing_snapshots,
      snapshot_checker, vstorage, log_buffer, full_history_ts_low,
      require_max_output_level);
  NoteCompactionStartLevelIfNeeded(c);
  return c;
}

Compaction* CustomCompactionPicker::TryCustomCompactionForLevel(
    int start_level, int /*output_level*/, VersionStorageInfo* vstorage,
    const std::string& cf_name, const MutableCFOptions& mutable_cf_options,
    const MutableDBOptions& mutable_db_options, LogBuffer* log_buffer,
    const std::string& full_history_ts_low, double start_level_score,
    std::string* normal_reason) {
  if (vstorage == nullptr) {
    if (normal_reason) *normal_reason = "vstorage_null";
    return nullptr;
  }
  if (start_level == 0) {
    if (normal_reason) *normal_reason = "level_zero";
    return nullptr;
  }
  if (!g_two_phase_write_manager || !g_two_phase_write_manager->IsInitialized() ||
      !g_two_phase_write_manager->IsPhase2Enabled()) {
    if (normal_reason) *normal_reason = "use_custom_disabled";
    return nullptr;
  }
  if (g_custom_compaction_pri_manager == nullptr ||
      !g_custom_compaction_pri_manager->ShouldUseCustomPri(start_level)) {
    if (normal_reason) *normal_reason = "use_custom_disabled";
    return nullptr;
  }
  CustomReservationResult reserve =
      g_two_phase_write_manager->TryReserveCustomForLevel(start_level);
  if (reserve == CustomReservationResult::kBlocked) {
    if (normal_reason) *normal_reason = "budget_cap_reached";
    return nullptr;
  }
  const bool need_release_on_fail =
      (reserve == CustomReservationResult::kAcquired);
  std::vector<FileLifetimeInfo> too_far =
      g_two_phase_write_manager->GetAllTooFarFiles(start_level, vstorage);
  if (too_far.empty()) {
    if (need_release_on_fail) {
      g_two_phase_write_manager->ReleaseCustomReservation(start_level);
    }
    if (normal_reason) *normal_reason = "no_too_far_files";
    if (ioptions_.info_log) {
      ROCKS_LOG_INFO(ioptions_.info_log,
                     "[CustomCompactionPicker] level=%d: no too-far files, use normal pick",
                     start_level);
    }
    return nullptr;
  }
  static const std::vector<SequenceNumber> kEmptySnapshots;
  for (const FileLifetimeInfo& info : too_far) {
    Compaction* c = SelectFileFromPriorityQueue(
        cf_name, mutable_cf_options, mutable_db_options, kEmptySnapshots,
        nullptr, vstorage, log_buffer, full_history_ts_low, info.file_number,
        start_level, CompactionReason::kLevelTooFarFiles, start_level_score);
    if (c != nullptr) {
      if (ioptions_.info_log) {
        ROCKS_LOG_INFO(ioptions_.info_log,
                       "[CustomCompactionPicker] level=%d: use custom compaction from too-far file #%" PRIu64,
                       start_level, info.file_number);
      }
      return c;
    }
  }
  if (need_release_on_fail) {
    g_two_phase_write_manager->ReleaseCustomReservation(start_level);
  }
  if (normal_reason) *normal_reason = "all_select_failed";
  if (ioptions_.info_log) {
    ROCKS_LOG_INFO(ioptions_.info_log,
                   "[CustomCompactionPicker] level=%d: too-far count=%zu but all SelectFileFromPriorityQueue failed, use normal pick",
                   start_level, too_far.size());
  }
  return nullptr;
}

Compaction* CustomCompactionPicker::SelectFileFromPriorityQueue(
    const std::string& cf_name, const MutableCFOptions& mutable_cf_options,
    const MutableDBOptions& mutable_db_options,
    const std::vector<SequenceNumber>& /*existing_snapshots*/,
    const SnapshotChecker* /*snapshot_checker*/,
    VersionStorageInfo* vstorage, LogBuffer* /*log_buffer*/,
    const std::string& full_history_ts_low,
    uint64_t file_number, int level,
    CompactionReason compaction_reason, double compaction_score) {


  FileMetaData* file_meta = nullptr;
  int actual_level = level;
  for (int l = 0; l < vstorage->num_levels(); ++l) {
    const std::vector<FileMetaData*>& files = vstorage->LevelFiles(l);
    for (FileMetaData* f : files) {
      if (f->fd.GetNumber() == file_number) {
        file_meta = f;
        actual_level = l;
        break;
      }
    }
    if (file_meta != nullptr) {
      break;
    }
  }

  if (file_meta == nullptr) {
    if (ioptions_.info_log) {
      ROCKS_LOG_ERROR(ioptions_.info_log,
                     "[CustomCompactionPicker] ERROR: file #%" PRIu64 " not found (level %d)",
                     file_number, level);
    }
    return nullptr;
  }


  if (file_meta->being_compacted) {
    if (ioptions_.info_log) {
      ROCKS_LOG_INFO(ioptions_.info_log,
                     "[CustomCompactionPicker] WARNING: file #%" PRIu64 " is being compacted (level %d)",
                     file_number, actual_level);
    }
    return nullptr;
  }


  int output_level = (actual_level == 0) ? vstorage->base_level() : actual_level + 1;
  if (output_level >= vstorage->num_levels()) {
    output_level = vstorage->num_levels() - 1;
  }



  CompactionInputFiles start_level_inputs;
  start_level_inputs.level = actual_level;
  start_level_inputs.files.push_back(file_meta);



  if (!ExpandInputsToCleanCut(cf_name, vstorage,
                              &start_level_inputs)) {
    if (ioptions_.info_log) {
      ROCKS_LOG_INFO(ioptions_.info_log,
                     "[CustomCompactionPicker] ExpandInputsToCleanCut failed: file #%" PRIu64 " level=%d",
                     file_number, actual_level);
    }
    return nullptr;
  }


  CompactionInputFiles output_level_inputs;
  output_level_inputs.level = output_level;
  int parent_index = -1;
  int base_index = -1;








  bool is_l0_trivial_move = false;
  if (actual_level == 0 && output_level != 0 && !is_l0_trivial_move) {

    if (!GetOverlappingL0Files(vstorage, &start_level_inputs, output_level, &parent_index)) {
      if (ioptions_.info_log) {
        ROCKS_LOG_INFO(ioptions_.info_log,
                       "[CustomCompactionPicker] GetOverlappingL0Files failed: file #%" PRIu64 " level=%d->%d",
                       file_number, actual_level, output_level);
      }
      return nullptr;
    }
  }

  if (output_level != 0) {


    InternalKey smallest, largest;
    GetRange(start_level_inputs, &smallest, &largest);
    std::vector<FileMetaData*> overlapping_files;
    int temp_parent_index = -1;
    vstorage->GetOverlappingInputs(output_level, &smallest, &largest,
                                   &overlapping_files, -1, &temp_parent_index);


    if (AreFilesInCompaction(overlapping_files)) {
      if (ioptions_.info_log) {
        ROCKS_LOG_INFO(ioptions_.info_log,
                       "[CustomCompactionPicker] overlapping next-level file is being compacted: file #%" PRIu64 " level=%d->%d, output_level_files=%zu",
                       file_number, actual_level, output_level, overlapping_files.size());
      }
      return nullptr;
    }



    bool round_robin_expanding =
        ioptions_.compaction_pri == kRoundRobin &&
        (compaction_reason == CompactionReason::kLevelMaxLevelSize ||
         compaction_reason == CompactionReason::kLevelTooFarFiles);





    if (!SetupOtherInputs(cf_name, mutable_cf_options, vstorage,
                          &start_level_inputs, &output_level_inputs,
                          &parent_index, base_index, round_robin_expanding)) {


      CompactionInputFiles temp_output_inputs;
      temp_output_inputs.level = output_level;
      InternalKey temp_smallest, temp_largest;
      GetRange(start_level_inputs, &temp_smallest, &temp_largest);
      int temp_parent_index2 = -1;
      vstorage->GetOverlappingInputs(output_level, &temp_smallest, &temp_largest,
                                     &temp_output_inputs.files, -1,
                                     &temp_parent_index2);


      bool has_files_in_compaction = AreFilesInCompaction(temp_output_inputs.files);

      if (ioptions_.info_log) {
        ROCKS_LOG_INFO(ioptions_.info_log,
                       "[CustomCompactionPicker] SetupOtherInputs failed: file #%" PRIu64 " level=%d->%d, output_level_files=%zu, has_files_in_compaction=%d",
                       file_number, actual_level, output_level, temp_output_inputs.files.size(), has_files_in_compaction ? 1 : 0);
      }
      return nullptr;
    }



    std::vector<CompactionInputFiles> temp_compaction_inputs = {start_level_inputs};
    if (!output_level_inputs.empty()) {
      temp_compaction_inputs.push_back(output_level_inputs);
    }


    int proximal_level = Compaction::EvaluateProximalLevel(vstorage, mutable_cf_options,
                                                           ioptions_, actual_level,
                                                           output_level);
    if (FilesRangeOverlapWithCompaction(temp_compaction_inputs, output_level, proximal_level)) {
      if (ioptions_.info_log) {
        ROCKS_LOG_INFO(ioptions_.info_log,
                       "[CustomCompactionPicker] overlaps an active compaction: file #%" PRIu64 " (level %d->%d)",
                       file_number, actual_level, output_level);
      }
      return nullptr;
    }



  } else {


  }




  if (start_level_inputs.files.empty()) {
    if (ioptions_.info_log) {
      ROCKS_LOG_WARN(ioptions_.info_log,
                     "[CustomCompactionPicker] start_level_inputs is empty; returning nullptr: file #%" PRIu64 " level=%d",
                     file_number, actual_level);
    }
    return nullptr;
  }
  std::vector<CompactionInputFiles> compaction_inputs = {start_level_inputs};
  if (!output_level_inputs.empty()) {
    compaction_inputs.push_back(output_level_inputs);
  }

  std::vector<FileMetaData*> grandparents;



  if (output_level != 0 && !is_l0_trivial_move) {
    GetGrandparents(vstorage, start_level_inputs, output_level_inputs, &grandparents);
  }


  uint64_t max_file_size = MaxFileSizeForLevel(
      mutable_cf_options, output_level, ioptions_.compaction_style,
      vstorage->base_level(), ioptions_.level_compaction_dynamic_level_bytes);



  uint32_t path_id = 0;


  if (!ioptions_.cf_paths.empty()) {
    uint32_t p = 0;
    // size remaining in the most recent path
    uint64_t current_path_size = ioptions_.cf_paths[0].target_size;

    uint64_t level_size;
    int cur_level = 0;

    // max_bytes_for_level_base denotes L1 size.
    // We estimate L0 size to be the same as L1.
    level_size = mutable_cf_options.max_bytes_for_level_base;

    // Last path is the fallback
    while (p < ioptions_.cf_paths.size() - 1) {
      if (level_size <= current_path_size) {
        if (cur_level == output_level) {
          // Does desired level fit in this path?
          path_id = p;
          break;
        } else {
          current_path_size -= level_size;
          if (cur_level > 0) {
            if (ioptions_.level_compaction_dynamic_level_bytes) {
              // Currently, level_compaction_dynamic_level_bytes is ignored when
              // multiple db paths are specified. https://github.com/facebook/
              // rocksdb/blob/main/db/column_family.cc.
              // Still, adding this check to avoid accidentally using
              // max_bytes_for_level_multiplier_additional
              level_size = static_cast<uint64_t>(
                  level_size * mutable_cf_options.max_bytes_for_level_multiplier);
            } else {
              level_size = static_cast<uint64_t>(
                  level_size * mutable_cf_options.max_bytes_for_level_multiplier *
                  mutable_cf_options.MaxBytesMultiplerAdditional(cur_level));
            }
          }
          cur_level++;
          continue;
        }
      }
      p++;


      current_path_size = ioptions_.cf_paths[p].target_size;
    }

    path_id = p;
  }


  CompressionType compression_type = GetCompressionType(
      vstorage, mutable_cf_options, output_level, vstorage->base_level());
  CompressionOptions compression_opts = GetCompressionOptions(
      mutable_cf_options, vstorage, output_level);









  if (compaction_inputs.empty()) {
    if (ioptions_.info_log) {
      ROCKS_LOG_WARN(ioptions_.info_log,
                     "[CustomCompactionPicker] compaction_inputs is empty; returning nullptr: file #%" PRIu64 " level=%d",
                     file_number, actual_level);
    }
    return nullptr;
  }
  bool l0_files_might_overlap = (actual_level == 0 && !is_l0_trivial_move &&
                                  (compaction_inputs.size() > 1 || compaction_inputs[0].size() > 1));

  auto c = new Compaction(
      vstorage, ioptions_, mutable_cf_options, mutable_db_options,
      std::move(compaction_inputs), output_level,
      max_file_size, mutable_cf_options.max_compaction_bytes,
      path_id, compression_type, compression_opts,
      Temperature::kUnknown,
      /* max_subcompactions */ 0, std::move(grandparents),
      /* earliest_snapshot */ std::nullopt, /* snapshot_checker */ nullptr,
      compaction_reason,
      /* trim_ts */ "", compaction_score, l0_files_might_overlap);


  RegisterCompaction(c);


  vstorage->ComputeCompactionScore(ioptions_, mutable_cf_options,
                                   full_history_ts_low);

  if (ioptions_.info_log) {
    ROCKS_LOG_INFO(ioptions_.info_log,
                   "[CustomCompactionPicker] Created compaction: level %d -> %d, file #%" PRIu64
                   " (input files: %zu, output level files: %zu, grandparents: %zu)",
                   actual_level, output_level, file_number,
                   start_level_inputs.files.size(),
                   output_level_inputs.files.size(),
                   grandparents.size());
  }

  return c;
}

}  // namespace ROCKSDB_NAMESPACE
