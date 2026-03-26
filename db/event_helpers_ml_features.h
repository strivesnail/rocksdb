#pragma once

#include <cstddef>
#include <vector>
#include "rocksdb/env.h"
#include "db/dbformat.h"
#include "db/version_edit.h"

namespace ROCKSDB_NAMESPACE {

class ColumnFamilyData;
class InstrumentedMutex;

// MLFeatures structure definition (69 features total)
// All features are included for compatibility with 69-feature models
struct MLFeatures {
  // Rank and score features (11 features, indices 0-10)
  int first_active_rank;           // 0
  int first_passive_rank;          // 1
  double active_rank_normalized;   // 2
  double passive_rank_normalized;  // 3
  double file_count_ratio;         // 4
  double first_active_score;       // 5
  double first_passive_score;      // 6
  double level_avg_score;          // 7
  double score_rank_interaction;   // 8
  int rank_difference;             // 9
  double score_difference;         // 10
  
  // Key range features (6 features, indices 11-16)
  uint64_t key_range_start;        // 11
  uint64_t key_range_end;          // 12
  uint64_t key_range_size;         // 13
  double log10_key_range_size;     // 14
  double key_range_position_in_level;    // 15
  double key_range_percentile_in_level;  // 16
  
  // Level stats features (35 features, indices 17-51)
  // Level 0
  uint64_t level0_current_file_count;           // 17
  uint64_t level0_cumulative_compaction_count;  // 18
  uint64_t level0_cumulative_file_count;        // 19
  
  // Level 1
  uint64_t level1_current_file_count;           // 20
  uint64_t level1_cumulative_compaction_count;  // 21
  uint64_t level1_cumulative_file_count;        // 22
  uint64_t level1_cumulative_trivial_move_count;// 23
  uint64_t level1_total_size;                   // 24
  
  // Level 2
  uint64_t level2_current_file_count;           // 25
  uint64_t level2_cumulative_compaction_count;  // 26
  uint64_t level2_cumulative_file_count;        // 27
  uint64_t level2_cumulative_trivial_move_count;// 28
  uint64_t level2_total_size;                   // 29
  
  // Level 3
  uint64_t level3_current_file_count;           // 30
  uint64_t level3_cumulative_compaction_count;  // 31
  uint64_t level3_cumulative_file_count;        // 32
  uint64_t level3_cumulative_trivial_move_count;// 33
  uint64_t level3_total_size;                   // 34
  
  // Level 4
  uint64_t level4_current_file_count;           // 35
  uint64_t level4_cumulative_compaction_count;  // 36
  uint64_t level4_cumulative_file_count;        // 37
  uint64_t level4_cumulative_trivial_move_count;// 38
  uint64_t level4_total_size;                   // 39
  
  // Level 5
  uint64_t level5_current_file_count;           // 40
  uint64_t level5_cumulative_compaction_count;  // 41
  uint64_t level5_cumulative_file_count;        // 42
  uint64_t level5_cumulative_trivial_move_count;// 43
  uint64_t level5_total_size;                   // 44
  
  // Level 6
  uint64_t level6_current_file_count;           // 45
  uint64_t level6_cumulative_compaction_count;  // 46
  uint64_t level6_cumulative_file_count;        // 47
  uint64_t level6_cumulative_trivial_move_count;// 48
  uint64_t level6_total_size;                   // 49
  
  // Overlap features (6 features, indices 50-55)
  uint64_t overlap_with_lower;       // 50
  uint64_t overlap_with_upper;       // 51
  uint64_t overlap_count_with_lower; // 52
  uint64_t overlap_count_with_upper; // 53
  double overlap_ratio_with_lower;   // 54
  double overlap_ratio_with_upper;   // 55
  
  // Neighbor distance features (4 features, indices 56-59)
  double left_neighbor_key_distance;  // 56
  double right_neighbor_key_distance; // 57
  double min_neighbor_key_distance;   // 58
  double avg_neighbor_key_distance;   // 59
  
  // Competition features (4 features, indices 60-63)
  uint64_t better_score_files_count;  // 60
  uint64_t worse_score_files_count;   // 61
  double competition_ratio;           // 62
  uint64_t neighbor_files_count;      // 63
  
  // Other features (5 features, indices 64-68)
  double lower_level_capacity_ratio;  // 64
  double upper_level_capacity_ratio;  // 65
  double urgency_score;               // 66
  double health_score;                // 67
  double stability_score;             // 68
  
  // Creation level (not in the 69 features array, but used internally)
  int creation_level;
  
  // Key range string (for logging only, not used in ML training)
  std::string key_range;
};

// Total: 69 features
// - Rank/Score: 11 (indices 0-10)
// - Key Range: 6 (indices 11-16)
// - Level Stats: 33 (indices 17-49)
// - Overlap: 6 (indices 50-55)
// - Neighbor: 4 (indices 56-59)
// - Competition: 4 (indices 60-63)
// - Other: 5 (indices 64-68)

// Map predicted lifetime to WriteLifeTimeHint.
Env::WriteLifeTimeHint MapLifetimeToHint(double predicted_lifetime, int level);

// Convert MLFeatures to 69-element double array (matching training order)
void MLFeaturesToArray(const MLFeatures& features, double* features_array, size_t array_size);

// Calculate ML features for a file
bool CalculateMLFeatures(const FileDescriptor& fd, const InternalKey& smallest,
                         const InternalKey& largest, uint64_t compensated_file_size,
                         uint64_t num_entries, int level, ColumnFamilyData* cfd,
                         MLFeatures* features, InstrumentedMutex* db_mutex);

// Calculate ML features before file write (file_size not available)
// Uses estimated file size based on num_entries or estimated_file_size
// Parameters:
//   - smallest, largest, num_entries, estimated_file_size, level, cfd, features, db_mutex: as before
//   - subset_indices: optional; when non-null and non-empty, only compute feature groups that
//     intersect this set (saves cost). Groups: 0-10 rank, 11-16 key, 17-49 level_stats,
//     50-55 overlap, 56-59 neighbor, 60-63 competition, 64-68 other.
bool CalculateMLFeaturesBeforeWrite(const InternalKey& smallest,
                                    const InternalKey& largest,
                                    uint64_t num_entries,
                                    uint64_t estimated_file_size,
                                    int level,
                                    ColumnFamilyData* cfd,
                                    MLFeatures* features,
                                    InstrumentedMutex* db_mutex,
                                    const std::vector<int>* subset_indices = nullptr);

// Convert user key to numeric (for binary keys: big-endian uint64)
uint64_t KeyToNumeric(const Slice& user_key);

// Convert uint64 to 8-byte big-endian key
std::string NumericToKeyBigEndian(uint64_t val);

// Estimate largest user key for output file at current_file_index when total
// range is (range_start, range_end) and there are estimated_output_files.
// Returns empty string if keys are not 8-byte binary (caller should use fallback).
std::string EstimateFileLargestKey(const Slice& range_start,
                                   const Slice& range_end,
                                   int estimated_output_files,
                                   int current_file_index);

}  // namespace ROCKSDB_NAMESPACE
