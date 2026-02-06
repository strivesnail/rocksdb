#pragma once

#include <cstddef>
#include "rocksdb/env.h"
#include "db/dbformat.h"
#include "db/version_edit.h"

namespace ROCKSDB_NAMESPACE {

class ColumnFamilyData;
class InstrumentedMutex;

// MLFeatures structure definition (69 features total)
struct MLFeatures {
  // Rank and score features (0-9)
  int first_active_rank;
  int first_passive_rank;
  double active_rank_normalized;
  double passive_rank_normalized;
  double file_count_ratio;
  double first_active_score;
  double first_passive_score;
  double level_avg_score;
  double score_rank_interaction;
  double rank_difference;
  double score_difference;
  
  // Key range features (11-16)
  uint64_t key_range_start;
  uint64_t key_range_end;
  uint64_t key_range_size;
  double log10_key_range_size;
  double key_range_position_in_level;
  double key_range_percentile_in_level;
  
  // Level 0 features (17-19)
  uint64_t level0_current_file_count;
  uint64_t level0_cumulative_compaction_count;
  uint64_t level0_cumulative_file_count;
  
  // Level 1 features (20-24)
  uint64_t level1_current_file_count;
  uint64_t level1_cumulative_compaction_count;
  uint64_t level1_cumulative_file_count;
  uint64_t level1_cumulative_trivial_move_count;
  uint64_t level1_total_size;
  
  // Level 2 features (25-29)
  uint64_t level2_current_file_count;
  uint64_t level2_cumulative_compaction_count;
  uint64_t level2_cumulative_file_count;
  uint64_t level2_cumulative_trivial_move_count;
  uint64_t level2_total_size;
  
  // Level 3 features (30-34)
  uint64_t level3_current_file_count;
  uint64_t level3_cumulative_compaction_count;
  uint64_t level3_cumulative_file_count;
  uint64_t level3_cumulative_trivial_move_count;
  uint64_t level3_total_size;
  
  // Level 4 features (35-39)
  uint64_t level4_current_file_count;
  uint64_t level4_cumulative_compaction_count;
  uint64_t level4_cumulative_file_count;
  uint64_t level4_cumulative_trivial_move_count;
  uint64_t level4_total_size;
  
  // Level 5 features (40-44)
  uint64_t level5_current_file_count;
  uint64_t level5_cumulative_compaction_count;
  uint64_t level5_cumulative_file_count;
  uint64_t level5_cumulative_trivial_move_count;
  uint64_t level5_total_size;
  
  // Level 6 features (45-49)
  uint64_t level6_current_file_count;
  uint64_t level6_cumulative_compaction_count;
  uint64_t level6_cumulative_file_count;
  uint64_t level6_cumulative_trivial_move_count;
  uint64_t level6_total_size;
  
  // Overlap features (50-55)
  uint64_t overlap_with_lower;
  uint64_t overlap_with_upper;
  uint64_t overlap_count_with_lower;
  uint64_t overlap_count_with_upper;
  double overlap_ratio_with_lower;
  double overlap_ratio_with_upper;
  
  // Neighbor distance features (56-59)
  double left_neighbor_key_distance;
  double right_neighbor_key_distance;
  double min_neighbor_key_distance;
  double avg_neighbor_key_distance;
  
  // Competition features (60-63)
  uint64_t better_score_files_count;
  uint64_t worse_score_files_count;
  double competition_ratio;
  uint64_t neighbor_files_count;
  
  // Capacity and score features (64-68)
  double lower_level_capacity_ratio;
  double upper_level_capacity_ratio;
  double urgency_score;
  double health_score;
  double stability_score;
  
  // Creation level (not in the 69 features array, but used internally)
  int creation_level;
  
  // Key range string (for logging only, not used in ML training)
  std::string key_range;
};

// Map predicted lifetime to WriteLifeTimeHint.
Env::WriteLifeTimeHint MapLifetimeToHint(double predicted_lifetime, int level);

// Convert MLFeatures to 69-element double array (matching training order)
void MLFeaturesToArray(const MLFeatures& features, double* features_array, size_t array_size);

// Calculate ML features for a file
bool CalculateMLFeatures(const FileDescriptor& fd, const InternalKey& smallest,
                         const InternalKey& largest, uint64_t compensated_file_size,
                         uint64_t num_entries, int level, ColumnFamilyData* cfd,
                         MLFeatures* features, InstrumentedMutex* db_mutex);

}  // namespace ROCKSDB_NAMESPACE
