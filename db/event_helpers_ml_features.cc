//  Copyright (c) 2011-present, Facebook, Inc.  All rights reserved.
//  This source code is licensed under both the GPLv2 (found in the
//  COPYING file in the root directory) and Apache 2.0 License
//  (found in the LICENSE.Apache file in the root directory).
//

#include "db/event_helpers_ml_features.h"

#include <algorithm>
#include <cmath>
#include <cctype>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <limits>
#include <mutex>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include "tools/ml_predict_python.h"

#ifdef ROCKSDB_ML_PREDICT_PYTHON
#include <Python.h>
#endif

#include "db/compaction/file_pri.h"
#include "db/dbformat.h"
#include "db/event_helpers.h"
#include "db/internal_stats.h"
#include "db/version_set.h"
#include "logging/event_logger.h"
#include "logging/logging.h"
#include "options/cf_options.h"
#include "port/port.h"
#include "rocksdb/env.h"

namespace ROCKSDB_NAMESPACE {

namespace {

#ifdef ROCKSDB_ML_PREDICT_PYTHON
static PyObject* g_ml_predict_module = nullptr;
static PyObject* g_predict_func = nullptr;
static std::once_flag g_python_init_flag;
static bool g_python_initialized = false;

static bool InitializePythonMLPredictor() {
  std::call_once(g_python_init_flag, []() {
    if (!Py_IsInitialized()) {
      Py_Initialize();
      if (!Py_IsInitialized()) {
        return;
      }
    }

    PyObject* sys_path = PySys_GetObject("path");
    const char* rocksdb_tools_path = std::getenv("ROCKSDB_TOOLS_PATH");
    if (!rocksdb_tools_path) {
      rocksdb_tools_path = "./tools";
    }
    PyObject* path = PyUnicode_FromString(rocksdb_tools_path);
    PyList_Append(sys_path, path);
    Py_DECREF(path);

    g_ml_predict_module = PyImport_ImportModule("ml_predict_lifetime_by_level");
    if (!g_ml_predict_module) {
      PyErr_Print();
      return;
    }

    g_predict_func = PyObject_GetAttrString(
        g_ml_predict_module, "predict_file_lifetime_by_level");
    if (!g_predict_func || !PyCallable_Check(g_predict_func)) {
      Py_XDECREF(g_predict_func);
      g_predict_func = nullptr;
      return;
    }

    PyObject* init_func =
        PyObject_GetAttrString(g_ml_predict_module, "initialize");
    bool init_success = false;
    if (init_func && PyCallable_Check(init_func)) {
      PyObject* result = PyObject_CallObject(init_func, nullptr);
      if (result) {
        init_success = PyObject_IsTrue(result);
        Py_DECREF(result);
      } else {
        if (PyErr_Occurred()) {
          PyErr_Clear();
        }
        init_success = false;
      }
      Py_DECREF(init_func);
    } else {
      init_success = (g_predict_func != nullptr);
    }

    g_python_initialized = (g_predict_func != nullptr) && init_success;

    if (g_python_initialized) {
      PyEval_SaveThread();
    }
  });

  return g_python_initialized;
}
#endif

// Helper function to convert InternalKey to numeric value (delegates to KeyToNumeric(Slice))
uint64_t KeyToNumeric(const InternalKey& ikey) {
  if (ikey.unset() || ikey.size() == 0) {
    return 0;
  }
  return KeyToNumeric(ikey.user_key());
}

// Helper function to calculate file score for kMinOverlappingRatio.
// Uses VersionStorageInfo::ComputeOverlappingBytesWithLevel for exact consistency
// with RocksDB native SortFileByOverlappingRatio.
double CalculateFileScoreForMinOverlappingRatio(
    const InternalKey* file_smallest, const InternalKey* file_largest,
    uint64_t file_number, uint64_t compensated_file_size,
    VersionStorageInfo* vstorage, int level,
    const InternalKeyComparator* icmp, const ImmutableOptions& ioptions,
    const MutableCFOptions& mutable_cf_options, Logger* info_log,
    InstrumentedMutex* db_mutex) {
  (void)file_number;
  (void)compensated_file_size;
  (void)ioptions;
  (void)mutable_cf_options;
  (void)info_log;
  (void)db_mutex;
  if (level >= vstorage->num_levels() - 1) {
    return 0.0;  // Last level has no next level
  }
  uint64_t overlapping_bytes = vstorage->ComputeOverlappingBytesWithLevel(
      *file_smallest, *file_largest, level + 1, *icmp);
  return static_cast<double>(overlapping_bytes);
}

// Helper function to calculate passive compaction score.
// Uses VersionStorageInfo::ComputeOverlappingBytesWithLevel for consistency
// with active score (same overlap logic as SortFileByOverlappingRatio).
static double CalculatePassiveScoreForMinOverlappingRatio(
    const InternalKey* file_smallest, const InternalKey* file_largest,
    uint64_t file_number, uint64_t compensated_file_size,
    VersionStorageInfo* vstorage, int level,
    const InternalKeyComparator* icmp, const ImmutableOptions& ioptions,
    const MutableCFOptions& mutable_cf_options, Logger* info_log,
    InstrumentedMutex* db_mutex) {
  (void)file_number;
  (void)compensated_file_size;
  (void)ioptions;
  (void)mutable_cf_options;
  (void)info_log;
  (void)db_mutex;
  if (level == 0) {
    return 0.0;  // Level0 doesn't have a previous level
  }
  uint64_t overlapping_bytes = vstorage->ComputeOverlappingBytesWithLevel(
      *file_smallest, *file_largest, level - 1, *icmp);
  return static_cast<double>(overlapping_bytes);
}

// CalculateOverlapRatio function removed - now using overlap_with_level / key_range_size
// for overlap_ratio_with_lower and overlap_ratio_with_upper calculations

// Calculate key distance between two files
// Returns positive value for gap, negative for overlap, 0 for adjacent
double CalculateKeyDistance(const FileMetaData* file1,
                              const FileMetaData* file2,
                              const InternalKeyComparator* icmp,
                              Logger* info_log) {
  // Safety check: report error for null or invalid file pointers
  if (file1 == nullptr || file2 == nullptr) {
    ROCKS_LOG_ERROR(info_log,
                    "[ML Features] Invalid null pointer in CalculateKeyDistance: "
                    "file1=%p, file2=%p",
                    static_cast<const void*>(file1),
                    static_cast<const void*>(file2));
    return std::numeric_limits<double>::max();
  }
  
  // Safely extract FileDescriptor and InternalKey values to local variables
  // This minimizes risk of accessing invalid memory
  FileDescriptor file1_fd, file2_fd;
  InternalKey file1_smallest, file1_largest, file2_smallest, file2_largest;
  
  try {
    // Copy FileDescriptor members individually
    file1_fd.packed_number_and_path_id = file1->fd.packed_number_and_path_id;
    file1_fd.file_size = file1->fd.file_size;
    file1_fd.smallest_seqno = file1->fd.smallest_seqno;
    file1_fd.largest_seqno = file1->fd.largest_seqno;
    file1_fd.table_reader = file1->fd.table_reader;
    
    file2_fd.packed_number_and_path_id = file2->fd.packed_number_and_path_id;
    file2_fd.file_size = file2->fd.file_size;
    file2_fd.smallest_seqno = file2->fd.smallest_seqno;
    file2_fd.largest_seqno = file2->fd.largest_seqno;
    file2_fd.table_reader = file2->fd.table_reader;
    
    // Copy InternalKey values
    file1_smallest = file1->smallest;
    file1_largest = file1->largest;
    file2_smallest = file2->smallest;
    file2_largest = file2->largest;
  } catch (...) {
    ROCKS_LOG_ERROR(info_log,
                    "[ML Features] Exception copying FileMetaData in "
                    "CalculateKeyDistance: file1_number=%" PRIu64
                    ", file2_number=%" PRIu64,
                    file1_fd.GetNumber(), file2_fd.GetNumber());
    return std::numeric_limits<double>::max();
  }
  
  if (file1_smallest.unset() || file1_largest.unset() ||
      file2_smallest.unset() || file2_largest.unset()) {
    ROCKS_LOG_ERROR(info_log,
                    "[ML Features] Invalid file with unset keys in "
                    "CalculateKeyDistance: file1_number=%" PRIu64
                    ", file2_number=%" PRIu64,
                    file1_fd.GetNumber(), file2_fd.GetNumber());
    return std::numeric_limits<double>::max();
  }
  
  int cmp1 = icmp->Compare(file1_largest, file2_smallest);
  int cmp2 = icmp->Compare(file1_smallest, file2_largest);

  if (cmp1 < 0) {
    // file1 is before file2, calculate gap: file2.key_start - file1.key_end
    uint64_t file1_end = KeyToNumeric(file1_largest);
    uint64_t file2_start = KeyToNumeric(file2_smallest);
    return static_cast<double>(file2_start) - static_cast<double>(file1_end);
  } else if (cmp2 > 0) {
    // file2 is before file1, calculate gap: file1.key_start - file2.key_end
    uint64_t file2_end = KeyToNumeric(file2_largest);
    uint64_t file1_start = KeyToNumeric(file1_smallest);
    return static_cast<double>(file1_start) - static_cast<double>(file2_end);
  } else {
    // Files overlap, return negative value (overlap amount)
    // Calculate overlap: min(file1_end, file2_end) - max(file1_start, file2_start)
    uint64_t file1_start = KeyToNumeric(file1_smallest);
    uint64_t file1_end = KeyToNumeric(file1_largest);
    uint64_t file2_start = KeyToNumeric(file2_smallest);
    uint64_t file2_end = KeyToNumeric(file2_largest);
    uint64_t overlap_start = std::max(file1_start, file2_start);
    uint64_t overlap_end = std::min(file1_end, file2_end);
    if (overlap_end >= overlap_start) {
      return -static_cast<double>(overlap_end - overlap_start + 1);
    } else {
      return 0.0;
    }
  }
}

}  // namespace

uint64_t KeyToNumeric(const Slice& user_key) {
  if (user_key.empty()) {
    return 0;
  }
  // For binary keys (common case: 8-byte uint64_t), interpret bytes as big-endian
  if (user_key.size() == sizeof(uint64_t)) {
    uint64_t result = 0;
    for (size_t i = 0; i < sizeof(uint64_t); ++i) {
      result = (result << 8) | static_cast<unsigned char>(user_key[i]);
    }
    return result;
  }
  if (user_key.size() <= sizeof(uint64_t)) {
    uint64_t result = 0;
    for (size_t i = 0; i < user_key.size(); ++i) {
      result = (result << 8) | static_cast<unsigned char>(user_key[i]);
    }
    return result;
  }
  // For string keys, try to parse as number or use hash
  std::string key_str = user_key.ToString();
  if (key_str.empty()) {
    return 0;
  }
  try {
    bool has_hex_chars = false;
    for (char c : key_str) {
      if ((c >= 'A' && c <= 'F') || (c >= 'a' && c <= 'f')) {
        has_hex_chars = true;
        break;
      }
    }
    uint64_t result = 0;
    if (has_hex_chars) {
      try {
        result = std::stoull(key_str, nullptr, 16);
      } catch (...) {
        if (std::isdigit(static_cast<unsigned char>(key_str[0]))) {
          result = std::stoull(key_str);
        } else {
          uint64_t hash_value = std::hash<std::string>{}(key_str);
          const uint64_t kMaxReasonableKey =
              static_cast<uint64_t>(std::numeric_limits<uint32_t>::max()) * 2;
          return hash_value % (kMaxReasonableKey + 1);
        }
      }
    } else if (std::isdigit(static_cast<unsigned char>(key_str[0]))) {
      result = std::stoull(key_str);
    } else {
      size_t pos = key_str.find_last_not_of("0123456789");
      if (pos != std::string::npos && pos + 1 < key_str.size()) {
        result = std::stoull(key_str.substr(pos + 1));
      } else {
        uint64_t hash_value = std::hash<std::string>{}(key_str);
        const uint64_t kMaxReasonableKey =
            static_cast<uint64_t>(std::numeric_limits<uint32_t>::max()) * 2;
        return hash_value % (kMaxReasonableKey + 1);
      }
    }
    const uint64_t kMaxReasonableKey =
        static_cast<uint64_t>(std::numeric_limits<uint32_t>::max()) * 2;
    return std::min(result, kMaxReasonableKey);
  } catch (...) {
    uint64_t hash_value = std::hash<std::string>{}(key_str);
    const uint64_t kMaxReasonableKey =
        static_cast<uint64_t>(std::numeric_limits<uint32_t>::max()) * 2;
    return hash_value % (kMaxReasonableKey + 1);
  }
}

std::string NumericToKeyBigEndian(uint64_t val) {
  std::string result(8, '\0');
  for (int i = 0; i < 8; ++i) {
    result[static_cast<size_t>(i)] =
        static_cast<char>((val >> (56 - 8 * i)) & 0xFF);
  }
  return result;
}

std::string EstimateFileLargestKey(const Slice& range_start,
                                   const Slice& range_end,
                                   int estimated_output_files,
                                   int current_file_index) {
  // Only support 8-byte binary keys for reversible conversion
  if (range_start.size() != sizeof(uint64_t) ||
      range_end.size() != sizeof(uint64_t)) {
    return "";
  }
  int N = std::max(1, estimated_output_files);
  uint64_t start_val = KeyToNumeric(range_start);
  uint64_t end_val = KeyToNumeric(range_end);
  if (end_val <= start_val) {
    return range_end.ToString();
  }
  uint64_t range_size = end_val - start_val;
  uint64_t segment = range_size / static_cast<uint64_t>(N);
  int i = std::max(0, current_file_index);
  if (i >= N) {
    return range_end.ToString();
  }
  uint64_t upper_val = start_val + static_cast<uint64_t>(i + 1) * segment;
  return NumericToKeyBigEndian(upper_val);
}

// Calculate ML features using safe value types (mimicking RocksDB native approach)
// Instead of passing FileMetaData* pointer, we pass individual fields as value types
// This avoids potential issues with invalid pointers, just like RocksDB does in
// LogAndNotifyTableFileCreationFinished which accepts const FileDescriptor& fd
bool CalculateMLFeatures(const FileDescriptor& fd, const InternalKey& smallest,
                         const InternalKey& largest, uint64_t compensated_file_size,
                         uint64_t num_entries, int level, ColumnFamilyData* cfd,
                         MLFeatures* features, InstrumentedMutex* db_mutex) {
  uint64_t file_number = fd.GetNumber();
  
  Logger* info_log = cfd ? cfd->ioptions().info_log.get() : nullptr;
  
  // Validate parameters (mimicking RocksDB's approach)
  if (cfd == nullptr || features == nullptr) {
    if (info_log) {
      ROCKS_LOG_ERROR(info_log,
                      "[ML Features] Invalid parameters: cfd=%p, features=%p",
                      static_cast<const void*>(cfd),
                      static_cast<const void*>(features));
    }
    return false;
  }

  // Only calculate for kMinOverlappingRatio
  if (cfd->ioptions().compaction_pri != kMinOverlappingRatio) {
    return false;
  }

  // Use value types directly (no pointer dereferencing needed)
  // This is exactly how RocksDB does it in event_helpers.cc:108-109
  // file_number already defined above for logging
  uint64_t file_size = fd.GetFileSize();
  InternalKey file_smallest = smallest;
  InternalKey file_largest = largest;

  // For level 0, skip ML features calculation entirely
  // Only log the original table file creation event without ML features
  if (level == 0) {
    return false;
  }

  // Initialize all features to 0
  *features = MLFeatures{};

  // Set creation_level (this is always available, even before other features are calculated)
  features->creation_level = level;

  // For levels > 0, need to access VersionStorageInfo for ML feature calculations
  // Check if file has valid keys first
  if (file_smallest.unset() || file_largest.unset()) {
    if (info_log) {
      ROCKS_LOG_WARN(info_log,
                     "[ML Features] File has unset keys: file_number=%" PRIu64
                     ", level=%d, smallest_unset=%d, largest_unset=%d",
                     file_number, level,
                     file_smallest.unset(), file_largest.unset());
    }
    return false;
  }

  // Use mutex to protect entire calculation process
  // This ensures data consistency and prevents concurrent modifications
  std::unique_ptr<InstrumentedMutexLock> lock;
  if (db_mutex != nullptr) {
    lock.reset(new InstrumentedMutexLock(db_mutex));
  }
  
  // Get Version and VersionStorageInfo under mutex protection
  Version* version = cfd->current();
  if (version == nullptr) {
    if (info_log) {
      ROCKS_LOG_ERROR(info_log,
                      "[ML Features] current() returned nullptr: file_number=%" PRIu64
                      ", level=%d",
                      file_number, level);
    }
    return false;
  }
  
  // Increment reference count to prevent Version from being deleted
  version->Ref();
  VersionStorageInfo* vstorage = version->storage_info();
  
  if (vstorage == nullptr) {
    version->Unref();
    if (info_log) {
      ROCKS_LOG_ERROR(info_log,
                      "[ML Features] vstorage is nullptr: file_number=%" PRIu64
                      ", level=%d",
                      file_number, level);
    }
    return false;
  }
  
  // RAII helper to ensure Version reference is released
  struct VersionRefGuard {
    Version* version_;
    
    VersionRefGuard(Version* v) : version_(v) {}
    
    ~VersionRefGuard() {
      if (version_ != nullptr) {
        version_->Unref();
      }
    }
    
    // Non-copyable
    VersionRefGuard(const VersionRefGuard&) = delete;
    VersionRefGuard& operator=(const VersionRefGuard&) = delete;
  };
  
  VersionRefGuard version_guard(version);
  
  const auto& ioptions = cfd->ioptions();
  const auto& mutable_cf_options = cfd->GetLatestMutableCFOptions();
  const InternalKeyComparator* icmp = &cfd->internal_comparator();
  // info_log already defined at function entry

  if (level < 0 || level >= vstorage->num_levels()) {
    if (info_log) {
      ROCKS_LOG_ERROR(info_log,
                      "[ML Features] Invalid level %d (num_levels=%d), file_number=%" PRIu64,
                      level, vstorage->num_levels(), file_number);
    }
    return false;
  }

  // Get current level files (excluding current file if not yet added)
  // Use NumLevelFiles() to safely get size, then copy files
  std::vector<FileMetaData*> level_files;
  try {
    // Use NumLevelFiles() to safely get file count first
    int level_files_count = vstorage->NumLevelFiles(level);
    if (level_files_count > 0) {
      const std::vector<FileMetaData*>& level_files_ref = vstorage->LevelFiles(level);
      
      // Sanity check: reasonable maximum (prevent invalid size values)
      const size_t kMaxReasonableFiles = 1000000;
      size_t level_files_size = static_cast<size_t>(level_files_count);
      if (level_files_size > kMaxReasonableFiles) {
        if (info_log) {
          ROCKS_LOG_ERROR(info_log,
                          "[ML Features] Invalid level_files size %zu (too large) at level %d, file_number=%" PRIu64,
                          level_files_size, level, file_number);
        }
        level_files_size = 0;
      }
      
      // Use size-based copy instead of iterator range to avoid iterator invalidation
      level_files.reserve(level_files_size);
      for (size_t i = 0; i < level_files_size; ++i) {
        const FileMetaData* f = nullptr;
        try {
          f = level_files_ref[i];
        } catch (...) {
          if (info_log) {
            ROCKS_LOG_ERROR(info_log,
                            "[ML Features] Failed to access level_files[%zu] at level %d, file_number=%" PRIu64,
                            i, level, file_number);
          }
          continue;
        }
        if (f != nullptr) {
          level_files.push_back(const_cast<FileMetaData*>(f));
        }
      }
    }
  } catch (...) {
    if (info_log) {
      ROCKS_LOG_ERROR(info_log,
                      "[ML Features] Exception getting level_files at level %d, file_number=%" PRIu64,
                      level, file_number);
    }
    // Continue with empty level_files
  }
  
  // Create a temporary file list that includes current file for score calculation
  std::vector<FileMetaData*> files_with_current = level_files;
  // Note: Current file might not be in vstorage yet, so we create a temporary
  // FileMetaData for calculation purposes
  FileMetaData temp_file;
  try {
    // Use value types directly (mimicking RocksDB native approach)
    temp_file.fd = fd;  // Use fd parameter directly (value type, safe)
    temp_file.smallest = file_smallest;
    temp_file.largest = file_largest;
    temp_file.compensated_file_size = compensated_file_size;
    temp_file.num_entries = num_entries;
    
    // Set oldest_ancester_time to current time for newly created files
    // This ensures ttl_boost_score is calculated correctly (should be 1 for new files)
    int64_t curr_time = 0;
    Status time_status = ioptions.clock->GetCurrentTime(&curr_time);
    if (time_status.ok()) {
      temp_file.oldest_ancester_time = static_cast<uint64_t>(curr_time);
    } else {
      // If GetCurrentTime fails, use 0 (kUnknownOldestAncesterTime)
      // This will cause ttl_boost_score to be large, but that's acceptable
      temp_file.oldest_ancester_time = 0;
    }
  } catch (...) {
    if (info_log) {
      ROCKS_LOG_ERROR(info_log,
                      "[ML Features] Exception creating temp_file at level %d, file_number=%" PRIu64,
                      level, file_number);
    }
    return false;
  }
  files_with_current.push_back(&temp_file);

  // ========== Score and Rank Features (1-9) ==========
  
  // Calculate active_score using extracted keys
  double active_score_raw = CalculateFileScoreForMinOverlappingRatio(
      &file_smallest, &file_largest, file_number, compensated_file_size, vstorage, level, icmp, ioptions,
      mutable_cf_options, info_log, db_mutex);
  
  // Debug logging: always log active_score_raw to diagnose score=0 issue
  if (info_log) {
    ROCKS_LOG_DEBUG(info_log,
                    "[ML Features] active_score_raw=%.0f for file_number=%" PRIu64 " at level=%d "
                    "(compensated_file_size=%" PRIu64 ")",
                    active_score_raw, file_number, level, compensated_file_size);
  }
  
  // Apply TTL boost and compensated_file_size normalization
  // Completely copy RocksDB native implementation from SortFileByOverlappingRatio
  uint64_t ttl = mutable_cf_options.ttl;
  int64_t curr_time;
  Status status = ioptions.clock->GetCurrentTime(&curr_time);
  if (!status.ok()) {
    // If we can't get time, disable TTL. (Exactly as RocksDB does)
    ttl = 0;
  }
  
  FileTtlBooster ttl_booster(static_cast<uint64_t>(curr_time), ttl,
                             vstorage->num_non_empty_levels(), level);
  uint64_t ttl_boost_score = (ttl > 0) ? ttl_booster.GetBoostScore(&temp_file) : 1;
  assert(ttl_boost_score > 0);
  
  // Check for division by zero - compensated_file_size should never be 0
  if (compensated_file_size == 0) {
    if (info_log) {
      ROCKS_LOG_ERROR(info_log,
                      "[ML Features] ERROR: compensated_file_size=0 at file_number=%" PRIu64
                      ", level=%d, file_size=%" PRIu64
                      " - cannot calculate score, returning false",
                      file_number, level, file_size);
    }
    fprintf(stderr, "[ML CalculateMLFeatures] ERROR: compensated_file_size=0, file_number=%" PRIu64 ", level=%d\n",
            file_number, level);
    fflush(stderr);
    return false;
  }
  
  // Calculate score exactly as RocksDB does: overlapping_bytes * 1024U / compensated_file_size / ttl_boost_score
  // This is integer division, producing uint64_t result
  uint64_t score = static_cast<uint64_t>(active_score_raw) * 1024U /
                   compensated_file_size /
                   ttl_boost_score;
  features->first_active_score = static_cast<double>(score);

  // Calculate first_passive_score
  double passive_score_raw = CalculatePassiveScoreForMinOverlappingRatio(
      &file_smallest, &file_largest, file_number, compensated_file_size, vstorage, level, icmp, ioptions,
      mutable_cf_options, info_log, db_mutex);
  
  // Debug logging: always log passive_score_raw to diagnose score=0 issue
  if (info_log) {
    ROCKS_LOG_DEBUG(info_log,
                    "[ML Features] passive_score_raw=%.0f for file_number=%" PRIu64 " at level=%d "
                    "(compensated_file_size=%" PRIu64 ")",
                    passive_score_raw, file_number, level, compensated_file_size);
  }
  
  // Calculate passive score exactly as active score (same TTL boost logic)
  // Completely copy RocksDB native implementation from SortFileByOverlappingRatio
  uint64_t passive_ttl = mutable_cf_options.ttl;
  int64_t passive_curr_time;
  Status passive_status = ioptions.clock->GetCurrentTime(&passive_curr_time);
  if (!passive_status.ok()) {
    // If we can't get time, disable TTL. (Exactly as RocksDB does)
    passive_ttl = 0;
  }
  
  FileTtlBooster passive_ttl_booster(static_cast<uint64_t>(passive_curr_time), passive_ttl,
                                    vstorage->num_non_empty_levels(), level);
  uint64_t passive_ttl_boost_score = (passive_ttl > 0) ? passive_ttl_booster.GetBoostScore(&temp_file) : 1;
  assert(passive_ttl_boost_score > 0);
  
  // Check for division by zero (compensated_file_size should already be checked, but be safe)
  if (compensated_file_size == 0) {
    if (info_log) {
      ROCKS_LOG_ERROR(info_log,
                      "[ML Features] ERROR: compensated_file_size=0 in passive score calculation at file_number=%" PRIu64
                      ", level=%d - returning false",
                      file_number, level);
    }
    fprintf(stderr, "[ML CalculateMLFeatures] ERROR: compensated_file_size=0 in passive score, file_number=%" PRIu64 ", level=%d\n",
            file_number, level);
    fflush(stderr);
    return false;
  }
  
  // Calculate score exactly as RocksDB does: overlapping_bytes * 1024U / compensated_file_size / ttl_boost_score
  // This is integer division, producing uint64_t result
  uint64_t passive_score_value = static_cast<uint64_t>(passive_score_raw) * 1024U /
                                  compensated_file_size /
                                  passive_ttl_boost_score;
  features->first_passive_score = static_cast<double>(passive_score_value);

  // Calculate scores for all files in level to determine rank
  // Optimization: Use a more efficient approach to calculate rank
  // Instead of calculating all scores and sorting, we can:
  // 1. Count how many files have lower scores than current file
  // 2. This avoids full sorting and reduces computation
  
  struct FileScoreInfo {
    const FileMetaData* file;
    double active_score;
    double passive_score;
  };
  
  // For large levels, limit the number of files we process to avoid slowdown
  const size_t max_files_for_rank_calc = 1000;
  const size_t files_to_process = std::min(level_files.size(), max_files_for_rank_calc);
  
  std::vector<FileScoreInfo> file_scores;
  file_scores.reserve(files_to_process + 1);

  // Calculate scores for existing files (limit to avoid slowdown)
  size_t processed = 0;
  for (const auto* f : level_files) {
    if (processed >= files_to_process) {
      break;
    }
    // Safety check: report error for null or invalid file pointers
    if (f == nullptr) {
      ROCKS_LOG_ERROR(info_log,
                      "[ML Features] Invalid null pointer in level_files "
                      "rank calculation at level %d, target_file_number=%" PRIu64,
                      level, file_number);
      continue;
    }
    if (f->smallest.unset() || f->largest.unset()) {
      ROCKS_LOG_ERROR(info_log,
                      "[ML Features] Invalid file with unset keys in "
                      "level_files rank calculation at level %d, "
                      "file_number=%" PRIu64 ", target_file_number=%" PRIu64,
                      level, f->fd.GetNumber(), file_number);
      continue;
    }
    // Extract fields to local variables (mimicking RocksDB native approach)
    // This avoids repeated pointer dereferencing and potential issues
    FileDescriptor f_fd = f->fd;  // Copy FileDescriptor (value type)
    InternalKey f_smallest = f->smallest;
    InternalKey f_largest = f->largest;
    uint64_t f_number = f_fd.GetNumber();  // Use copied fd
    uint64_t f_compensated_size = f->compensated_file_size;
    
    double f_active_score_raw = CalculateFileScoreForMinOverlappingRatio(
        &f_smallest, &f_largest, f_number, f_compensated_size, vstorage, level, icmp, ioptions,
        mutable_cf_options, info_log, db_mutex);
    
    // Apply TTL boost and normalization
    // Completely copy RocksDB native implementation from SortFileByOverlappingRatio
    uint64_t f_ttl = mutable_cf_options.ttl;
    int64_t f_curr_time;
    Status f_status = ioptions.clock->GetCurrentTime(&f_curr_time);
    if (!f_status.ok()) {
      // If we can't get time, disable TTL. (Exactly as RocksDB does)
      f_ttl = 0;
    }
    
    FileTtlBooster f_ttl_booster(static_cast<uint64_t>(f_curr_time), f_ttl,
                                vstorage->num_non_empty_levels(), level);
    uint64_t f_ttl_boost_score = (f_ttl > 0) ? f_ttl_booster.GetBoostScore(const_cast<FileMetaData*>(f)) : 1;
    assert(f_ttl_boost_score > 0);
    assert(f_compensated_size != 0);
    
    // Calculate score exactly as RocksDB does: overlapping_bytes * 1024U / compensated_file_size / ttl_boost_score
    uint64_t f_score = static_cast<uint64_t>(f_active_score_raw) * 1024U /
                      f_compensated_size /
                      f_ttl_boost_score;
    double active_score = static_cast<double>(f_score);
    
    // Calculate passive_score for each file
    double f_passive_score_raw = CalculatePassiveScoreForMinOverlappingRatio(
        &f_smallest, &f_largest, f_number, f_compensated_size, vstorage, level, icmp, ioptions,
        mutable_cf_options, info_log, db_mutex);
    
    // Calculate passive score exactly as active score (same TTL boost logic)
    // Completely copy RocksDB native implementation from SortFileByOverlappingRatio
    uint64_t f_passive_ttl = mutable_cf_options.ttl;
    int64_t f_passive_curr_time;
    Status f_passive_status = ioptions.clock->GetCurrentTime(&f_passive_curr_time);
    if (!f_passive_status.ok()) {
      // If we can't get time, disable TTL. (Exactly as RocksDB does)
      f_passive_ttl = 0;
    }
    
    FileTtlBooster f_passive_ttl_booster(static_cast<uint64_t>(f_passive_curr_time), f_passive_ttl,
                                         vstorage->num_non_empty_levels(), level);
    uint64_t f_passive_ttl_boost_score = (f_passive_ttl > 0) ? f_passive_ttl_booster.GetBoostScore(const_cast<FileMetaData*>(f)) : 1;
    assert(f_passive_ttl_boost_score > 0);
    assert(f_compensated_size != 0);
    
    // Calculate score exactly as RocksDB does: overlapping_bytes * 1024U / compensated_file_size / ttl_boost_score
    uint64_t f_passive_score = static_cast<uint64_t>(f_passive_score_raw) * 1024U /
                              f_compensated_size /
                              f_passive_ttl_boost_score;
    double passive_score = static_cast<double>(f_passive_score);
    file_scores.push_back({f, active_score, passive_score});
    processed++;
  }
  // Add current file (use temp_file for consistency)
  file_scores.push_back({&temp_file, features->first_active_score, features->first_passive_score});

  // Calculate active_rank by sorting files by score (and key as tiebreaker)
  // Sort in ascending order of score (lower score = better = lower rank)
  // This matches RocksDB's kMinOverlappingRatio strategy: prioritize files with less overlap
  // First prioritize marked_for_compaction=true files (exactly as RocksDB does)
  // When scores are equal, use key position as tiebreaker (smaller key = better = lower rank)
  std::sort(file_scores.begin(), file_scores.end(),
            [&](const FileScoreInfo& a, const FileScoreInfo& b) {
              // First compare by marked_for_compaction (exactly as RocksDB does)
              if (a.file->marked_for_compaction != b.file->marked_for_compaction) {
                return a.file->marked_for_compaction > b.file->marked_for_compaction;
              }
              // Then compare by score (lower is better for kMinOverlappingRatio)
              if (a.active_score != b.active_score) {
                return a.active_score < b.active_score;
              }
              // If scores are equal, compare by key (smaller is better)
              return icmp->Compare(a.file->smallest, b.file->smallest) < 0;
            });
  
  // Find the rank of the current file using standard competition ranking
  // Files with the same score get the same rank, next rank skips the tied count
  // Example: scores [10, 10, 5, 5, 0] -> ranks [1, 1, 3, 3, 5]
  int active_rank = 1;
  for (size_t i = 0; i < file_scores.size(); i++) {
    if (file_scores[i].file == &temp_file) {
      // Find the first file with the same score (going backwards)
      size_t first_same_score_idx = i;
      while (first_same_score_idx > 0 && 
             file_scores[first_same_score_idx - 1].active_score == file_scores[i].active_score) {
        first_same_score_idx--;
      }
      // Rank is the position of the first file with the same score + 1
      active_rank = static_cast<int>(first_same_score_idx + 1);
      
      // Debug logging: if score is 0 but rank is not 1, log why
      if (info_log && features->first_active_score == 0.0 && active_rank > 1) {
        // Count how many files have score > 0
        size_t files_with_higher_score = 0;
        for (size_t j = 0; j < file_scores.size(); j++) {
          if (file_scores[j].active_score > 0.0) {
            files_with_higher_score++;
          }
        }
        ROCKS_LOG_DEBUG(info_log,
                       "[ML Features] file_number=%" PRIu64 " at level=%d: "
                       "first_active_score=0 but first_active_rank=%d. "
                       "This is correct: %zu files have score > 0, "
                       "so rank %d means this file is the %d-th file with score=0 "
                       "(sorted by key as tiebreaker).",
                       file_number, level, active_rank,
                       files_with_higher_score, active_rank, active_rank - static_cast<int>(files_with_higher_score));
      }
      break;
    }
  }
  features->first_active_rank = active_rank;

  // Calculate passive_rank by sorting files by score (and key as tiebreaker)
  // Completely copy RocksDB native implementation from SortFileByOverlappingRatio
  // Sort in ascending order of score (lower score = better = lower rank)
  // First prioritize marked_for_compaction=true files (exactly as RocksDB does)
  std::sort(file_scores.begin(), file_scores.end(),
            [&](const FileScoreInfo& a, const FileScoreInfo& b) {
              // First compare by marked_for_compaction (exactly as RocksDB does)
              if (a.file->marked_for_compaction != b.file->marked_for_compaction) {
                return a.file->marked_for_compaction > b.file->marked_for_compaction;
              }
              // Then compare by score (lower is better, same as active_rank)
              if (a.passive_score != b.passive_score) {
                return a.passive_score < b.passive_score;
              }
              // If scores are equal, compare by key (smaller is better) - exactly as RocksDB does
              return icmp->Compare(a.file->smallest, b.file->smallest) < 0;
            });
  
  // Find the rank of the current file using standard competition ranking
  // Files with the same score get the same rank, next rank skips the tied count
  // Example: scores [10, 10, 5, 5, 0] -> ranks [1, 1, 3, 3, 5]
  int passive_rank = 1;
  for (size_t i = 0; i < file_scores.size(); i++) {
    if (file_scores[i].file == &temp_file) {
      // Find the first file with the same score (going backwards)
      size_t first_same_score_idx = i;
      while (first_same_score_idx > 0 && 
             file_scores[first_same_score_idx - 1].passive_score == file_scores[i].passive_score) {
        first_same_score_idx--;
      }
      // Rank is the position of the first file with the same score + 1
      passive_rank = static_cast<int>(first_same_score_idx + 1);
      break;
    }
  }
  features->first_passive_rank = passive_rank;

  // Calculate normalized ranks
  // Use NumLevelFiles() directly from RocksDB's data structure instead of level_files.size()
  // This ensures we include all files (including trivial move files) that are in vstorage
  // +1 because the current file hasn't been added to vstorage yet
  int current_file_count = vstorage->NumLevelFiles(level) + 1;
  if (current_file_count > 0) {
    features->active_rank_normalized =
        static_cast<double>(features->first_active_rank) / current_file_count;
    features->passive_rank_normalized =
        static_cast<double>(features->first_passive_rank) / current_file_count;
  }

  // Calculate interaction and difference features
  features->score_rank_interaction =
      features->first_active_score * features->first_active_rank;
  features->rank_difference = features->first_active_rank - features->first_passive_rank;
  features->score_difference = features->first_active_score - features->first_passive_score;

  // ========== Level State Features (10-37) ==========
  
  // Get cumulative_file_count from InternalStats (global counter, thread-safe, never reset)
  // This tracks all files ever created in the current level, including deleted ones
  InternalStats* internal_stats = cfd->internal_stats();
  uint64_t current_level_cumulative_file_count = 0;
  if (internal_stats != nullptr) {
    // TODO: GetCumulativeFileCount method not yet implemented in InternalStats
    // current_level_cumulative_file_count = internal_stats->GetCumulativeFileCount(level);
    current_level_cumulative_file_count = static_cast<uint64_t>(current_file_count);
    if (current_level_cumulative_file_count < static_cast<uint64_t>(current_file_count) &&
        info_log) {
      ROCKS_LOG_WARN(info_log,
                       "[ML Features] cumulative_file_count (%" PRIu64 ") < current_file_count (%d) for level %d. "
                     "file_number=%" PRIu64,
                       current_level_cumulative_file_count, current_file_count, level, file_number);
    }
  }
  // Note: cumulative_file_count for current level is stored in levelX_cumulative_file_count
  // No need to store it separately to avoid duplication

  // Calculate file_count_ratio
  // Level 0: ratio = current_file_count / level0_file_num_compaction_trigger
  // Level 1+: ratio = current_file_count if level_size >= MaxBytesForLevel, else 0
  if (level == 0) {
    int compaction_trigger = mutable_cf_options.level0_file_num_compaction_trigger;
    if (compaction_trigger > 0) {
      features->file_count_ratio =
          static_cast<double>(current_file_count) / compaction_trigger;
    }
  } else {
    // For level 1+, trigger is based on MaxBytesForLevel
    // We consider it triggered if level size >= MaxBytesForLevel
    uint64_t level_size = 0;
    for (const auto* f : level_files) {
      // Safety check: skip null or invalid file pointers
      if (f == nullptr) {
        ROCKS_LOG_ERROR(info_log,
                        "[ML Features] Invalid null pointer in level_files "
                        "file_count_ratio calculation at level %d, target_file_number=%" PRIu64,
                        level, file_number);
        continue;
      }
      // Extract FileDescriptor to local variable (mimicking RocksDB native approach)
      FileDescriptor f_fd = f->fd;  // Copy FileDescriptor (value type)
      level_size += f_fd.GetFileSize();  // Use copied fd
    }
    level_size += file_size;  // Use already extracted file_size instead of accessing file pointer
    if (level_size >= vstorage->MaxBytesForLevel(level)) {
      // Level is at capacity, ratio = current_file_count / 1 = current_file_count
      features->file_count_ratio = static_cast<double>(current_file_count);
    } else {
      // Level not at capacity, ratio = 0
      features->file_count_ratio = 0.0;
    }
  }

  // Calculate level_avg_score
  double sum_scores = 0.0;
  for (const auto& info : file_scores) {
    sum_scores += info.active_score;
  }
  if (file_scores.size() > 0) {
    features->level_avg_score = sum_scores / file_scores.size();
  }

  // Get file counts and total sizes for all levels (0-6)
  // Use NumLevelFiles() which is safer than accessing LevelFiles().size()
  for (int l = 0; l < 7 && l < vstorage->num_levels(); l++) {
    int count = 0;
    uint64_t total_size = 0;
    try {
      // Use NumLevelFiles() to safely get file count
      count = vstorage->NumLevelFiles(l);
      
      // Calculate total size for this level
      // Use NumLevelBytes() directly from RocksDB's data structure instead of manually iterating
      // This ensures we include all files (including trivial move files) and is more efficient
      if (count > 0) {
        try {
          total_size = vstorage->NumLevelBytes(l);
        } catch (...) {
          if (info_log) {
            ROCKS_LOG_DEBUG(info_log,
                            "[ML Features] Exception getting NumLevelBytes at level %d, target_file_number=%" PRIu64,
                            l, file_number);
          }
          // Fallback to 0 if NumLevelBytes fails
          total_size = 0;
        }
      }
    } catch (...) {
      if (info_log) {
        ROCKS_LOG_ERROR(info_log,
                        "[ML Features] Exception getting NumLevelFiles at level %d, target_file_number=%" PRIu64,
                        l, file_number);
      }
      // Continue with count = 0, total_size = 0
    }
    // Consistency: 0 files must imply 0 bytes
    if (count == 0) {
      total_size = 0;
    }
    switch (l) {
      case 0:
        features->level0_current_file_count = count;
        break;
      case 1:
        features->level1_current_file_count = count;
        features->level1_total_size = total_size;
        break;
      case 2:
        features->level2_current_file_count = count;
        features->level2_total_size = total_size;
        break;
      case 3:
        features->level3_current_file_count = count;
        features->level3_total_size = total_size;
        break;
      case 4:
        features->level4_current_file_count = count;
        features->level4_total_size = total_size;
        break;
      case 5:
        features->level5_current_file_count = count;
        features->level5_total_size = total_size;
        break;
      case 6:
        features->level6_current_file_count = count;
        features->level6_total_size = total_size;
        break;
    }
  }

  // Get cumulative compaction count and file count from RocksDB InternalStats
  // (comp_stats_[level].count, num_output_files, num_trivial_move_files)
  {
    const std::vector<InternalStats::CompactionStats>* comp_stats_ptr =
        (internal_stats != nullptr) ? &internal_stats->TEST_GetCompactionStats()
                                   : nullptr;
    for (int l = 0; l < 7; l++) {
      uint64_t compaction_count = 0;
      uint64_t cumulative_file_count = 0;
      uint64_t cumulative_trivial_move_count = 0;
      if (comp_stats_ptr != nullptr &&
          l < static_cast<int>(comp_stats_ptr->size())) {
        compaction_count = static_cast<uint64_t>((*comp_stats_ptr)[l].count);
        cumulative_file_count =
            static_cast<uint64_t>((*comp_stats_ptr)[l].num_output_files);
        cumulative_trivial_move_count =
            static_cast<uint64_t>((*comp_stats_ptr)[l].num_trivial_move_files);
      }
      switch (l) {
        case 0:
          features->level0_cumulative_compaction_count = compaction_count;
          features->level0_cumulative_file_count = cumulative_file_count;
          break;
        case 1:
          features->level1_cumulative_compaction_count = compaction_count;
          features->level1_cumulative_file_count = cumulative_file_count;
          features->level1_cumulative_trivial_move_count = cumulative_trivial_move_count;
          break;
        case 2:
          features->level2_cumulative_compaction_count = compaction_count;
          features->level2_cumulative_file_count = cumulative_file_count;
          features->level2_cumulative_trivial_move_count = cumulative_trivial_move_count;
          break;
        case 3:
          features->level3_cumulative_compaction_count = compaction_count;
          features->level3_cumulative_file_count = cumulative_file_count;
          features->level3_cumulative_trivial_move_count = cumulative_trivial_move_count;
          break;
        case 4:
          features->level4_cumulative_compaction_count = compaction_count;
          features->level4_cumulative_file_count = cumulative_file_count;
          features->level4_cumulative_trivial_move_count = cumulative_trivial_move_count;
          break;
        case 5:
          features->level5_cumulative_compaction_count = compaction_count;
          features->level5_cumulative_file_count = cumulative_file_count;
          features->level5_cumulative_trivial_move_count = cumulative_trivial_move_count;
          break;
        case 6:
          features->level6_cumulative_compaction_count = compaction_count;
          features->level6_cumulative_file_count = cumulative_file_count;
          features->level6_cumulative_trivial_move_count = cumulative_trivial_move_count;
          break;
      }
    }
  }

  // ========== Key Space Features (38-55) ==========
  
  // key_range: Format key range in hex format: "smallest .. largest"
  // Helper function to convert Slice to hex string
  auto SliceToHex = [](const Slice& s) -> std::string {
    std::ostringstream oss;
    oss << std::hex << std::uppercase << std::setfill('0');
    for (size_t i = 0; i < s.size(); ++i) {
      oss << std::setw(2) << static_cast<unsigned int>(static_cast<unsigned char>(s[i]));
    }
    return oss.str();
  };
  
  if (!file_smallest.unset() && !file_largest.unset()) {
    Slice smallest_user_key = file_smallest.user_key();
    Slice largest_user_key = file_largest.user_key();
    std::string smallest_hex = SliceToHex(smallest_user_key);
    std::string largest_hex = SliceToHex(largest_user_key);
    features->key_range = smallest_hex + " .. " + largest_hex;
    
    // key_range_size = largest_key - smallest_key + 1
    // Calculate directly from hex strings to avoid byte order issues
    // Also set key_range_start and key_range_end for ML training
    // If hex string is longer than 16 chars (64 bits), take only the last 16 chars (LSB)
    std::string smallest_hex_truncated = smallest_hex;
    std::string largest_hex_truncated = largest_hex;
    if (smallest_hex.size() > 16) {
      smallest_hex_truncated = smallest_hex.substr(smallest_hex.size() - 16);
    }
    if (largest_hex.size() > 16) {
      largest_hex_truncated = largest_hex.substr(largest_hex.size() - 16);
    }
    try {
      uint64_t smallest_numeric = std::stoull(smallest_hex_truncated, nullptr, 16);
      uint64_t largest_numeric = std::stoull(largest_hex_truncated, nullptr, 16);
      features->key_range_start = smallest_numeric;
      features->key_range_end = largest_numeric;
      if (largest_numeric >= smallest_numeric) {
        features->key_range_size = largest_numeric - smallest_numeric + 1;
      } else {
        features->key_range_size = 0;  // Invalid: largest < smallest
      }
    } catch (...) {
      features->key_range_start = KeyToNumeric(file_smallest);
      features->key_range_end = KeyToNumeric(file_largest);
      features->key_range_size = 0;  // Invalid: hex parse failed
    }
  } else {
    features->key_range = "N/A";
    features->key_range_start = KeyToNumeric(file_smallest);
    features->key_range_end = KeyToNumeric(file_largest);
    features->key_range_size = 0;  // Invalid: keys unset
  }
  features->log10_key_range_size =
      features->key_range_size > 0
          ? std::log10(static_cast<double>(features->key_range_size))
          : std::numeric_limits<double>::quiet_NaN();

  // Calculate overlap_with_lower
  // Use GetOverlappingInputs exactly as RocksDB does in compaction picker
  // Copy FileDescriptor values while holding mutex to avoid accessing freed memory
  if (level < vstorage->num_levels() - 1) {
    std::vector<FileDescriptor> lower_level_file_descriptors;
    try {
      // Validate level before calling
      int lower_level = level + 1;
      if (lower_level < 0 || lower_level >= vstorage->num_levels()) {
        if (info_log) {
          ROCKS_LOG_ERROR(info_log,
                          "[ML Features] Invalid lower_level %d (level=%d, num_levels=%d), file_number=%" PRIu64,
                          lower_level, level, vstorage->num_levels(), file_number);
        }
        // Continue with overlap_with_lower = 0
      } else {
        // Note: Mutex is already held by caller (CalculateMLFeatures)
        // No need to acquire mutex here
        
        // Get overlapping files
        std::vector<FileMetaData*> lower_level_files;
        lower_level_files.reserve(100);
        vstorage->GetOverlappingInputs(lower_level, &file_smallest, &file_largest,
                                        &lower_level_files);
        
        // Immediately copy FileDescriptor values while holding mutex
        // This prevents accessing freed FileMetaData objects after mutex is released
        lower_level_file_descriptors.reserve(lower_level_files.size());
        for (const FileMetaData* lower_file : lower_level_files) {
          if (lower_file == nullptr) {
            continue;
          }
          try {
            // Safely copy FileDescriptor by accessing members individually
            // This minimizes risk of accessing invalid memory
            FileDescriptor fd_copy;
            // Access fd members one by one to avoid accessing entire struct at once
            fd_copy.packed_number_and_path_id = lower_file->fd.packed_number_and_path_id;
            fd_copy.file_size = lower_file->fd.file_size;
            fd_copy.smallest_seqno = lower_file->fd.smallest_seqno;
            fd_copy.largest_seqno = lower_file->fd.largest_seqno;
            fd_copy.table_reader = lower_file->fd.table_reader;  // May be nullptr, that's OK
            lower_level_file_descriptors.push_back(fd_copy);
          } catch (...) {
            if (info_log) {
              ROCKS_LOG_ERROR(info_log,
                              "[ML Features] Exception copying FileDescriptor in "
                              "lower_level_files at level %d, file_number=%" PRIu64,
                              level + 1, file_number);
            }
            // Continue with next file
          }
        }
        // Note: Mutex is still held by caller (CalculateMLFeatures)
      }
    } catch (...) {
      if (info_log) {
        ROCKS_LOG_ERROR(info_log,
                        "[ML Features] Exception calling GetOverlappingInputs for lower level at level %d, file_number=%" PRIu64,
                        level + 1, file_number);
      }
      // Continue with overlap_with_lower = 0
    }
    
    // Now use copied FileDescriptor values, safe from concurrent modification
    for (const FileDescriptor& lower_fd : lower_level_file_descriptors) {
      try {
        features->overlap_with_lower += lower_fd.GetFileSize();
        features->overlap_count_with_lower++;
      } catch (...) {
        if (info_log) {
          ROCKS_LOG_ERROR(info_log,
                          "[ML Features] Exception getting file size from copied "
                          "FileDescriptor at level %d, file_number=%" PRIu64,
                          level + 1, file_number);
        }
        continue;
      }
    }
  }

  // Calculate overlap_with_upper
  // Use GetOverlappingInputs exactly as RocksDB does in compaction picker
  // Copy FileDescriptor values while holding mutex to avoid accessing freed memory
  if (level > 0) {
    std::vector<FileDescriptor> upper_level_file_descriptors;
    try {
      // Validate level before calling
      int upper_level = level - 1;
      if (upper_level < 0 || upper_level >= vstorage->num_levels()) {
        if (info_log) {
          ROCKS_LOG_ERROR(info_log,
                          "[ML Features] Invalid upper_level %d (level=%d, num_levels=%d), file_number=%" PRIu64,
                          upper_level, level, vstorage->num_levels(), file_number);
        }
        // Continue with overlap_with_upper = 0
      } else {
        // Note: Mutex is already held by caller (CalculateMLFeatures)
        // No need to acquire mutex here
        
        // Get overlapping files
        std::vector<FileMetaData*> upper_level_files;
        upper_level_files.reserve(100);
        vstorage->GetOverlappingInputs(upper_level, &file_smallest, &file_largest,
                                       &upper_level_files);
        
        // Immediately copy FileDescriptor values while holding mutex
        // This prevents accessing freed FileMetaData objects after mutex is released
        upper_level_file_descriptors.reserve(upper_level_files.size());
        for (const FileMetaData* upper_file : upper_level_files) {
          if (upper_file == nullptr) {
            continue;
          }
          try {
            // Safely copy FileDescriptor by accessing members individually
            // This minimizes risk of accessing invalid memory
            FileDescriptor fd_copy;
            // Access fd members one by one to avoid accessing entire struct at once
            fd_copy.packed_number_and_path_id = upper_file->fd.packed_number_and_path_id;
            fd_copy.file_size = upper_file->fd.file_size;
            fd_copy.smallest_seqno = upper_file->fd.smallest_seqno;
            fd_copy.largest_seqno = upper_file->fd.largest_seqno;
            fd_copy.table_reader = upper_file->fd.table_reader;  // May be nullptr, that's OK
            upper_level_file_descriptors.push_back(fd_copy);
          } catch (...) {
            if (info_log) {
              ROCKS_LOG_ERROR(info_log,
                              "[ML Features] Exception copying FileDescriptor in "
                              "upper_level_files at level %d, file_number=%" PRIu64,
                              level - 1, file_number);
            }
            // Continue with next file
          }
        }
        // Note: Mutex is still held by caller (CalculateMLFeatures)
      }
    } catch (...) {
      if (info_log) {
        ROCKS_LOG_ERROR(info_log,
                        "[ML Features] Exception calling GetOverlappingInputs for upper level at level %d, file_number=%" PRIu64,
                        level - 1, file_number);
      }
      // Continue with overlap_with_upper = 0
    }
    
    // Now use copied FileDescriptor values, safe from concurrent modification
    for (const FileDescriptor& upper_fd : upper_level_file_descriptors) {
      try {
        features->overlap_with_upper += upper_fd.GetFileSize();
        features->overlap_count_with_upper++;
      } catch (...) {
        if (info_log) {
          ROCKS_LOG_ERROR(info_log,
                          "[ML Features] Exception getting file size from copied "
                          "FileDescriptor at level %d, file_number=%" PRIu64,
                          level - 1, file_number);
        }
        continue;
      }
    }
  }

  // Note: file_size, log10_file_size, file_size_ratio, file_density, log10_file_density removed
  // These features are not collected before file write

  // Calculate overlap_ratio_with_lower
  // Formula: overlap_ratio = overlap_with_lower / key_range_size
  if (features->key_range_size > 0) {
    features->overlap_ratio_with_lower =
        static_cast<double>(features->overlap_with_lower) / features->key_range_size;
  }
  
  // Calculate overlap_ratio_with_upper
  // Formula: overlap_ratio = overlap_with_upper / key_range_size
  if (features->key_range_size > 0) {
    features->overlap_ratio_with_upper =
        static_cast<double>(features->overlap_with_upper) / features->key_range_size;
  }
  if (info_log) {
    ROCKS_LOG_DEBUG(info_log,
                    "[ML Features] Calculating overlap ratio: file_number=%" PRIu64
                    ", level=%d, key_range_size=%" PRIu64
                    ", overlap_with_lower=%" PRIu64
                    ", overlap_count_with_lower=%" PRIu64,
                    file_number, level,
                    features->key_range_size,
                    features->overlap_with_lower,
                    features->overlap_count_with_lower);
  }
  
  if (features->key_range_size > 0) {
    if (features->overlap_with_lower > 0) {
      features->overlap_ratio_with_lower =
          static_cast<double>(features->overlap_with_lower) /
          static_cast<double>(features->key_range_size);
      if (info_log) {
        ROCKS_LOG_DEBUG(info_log,
                        "[ML Features] Calculated overlap_ratio_with_lower: file_number=%" PRIu64
                        ", ratio=%.6f",
                        file_number, features->overlap_ratio_with_lower);
      }
    } else {
      features->overlap_ratio_with_lower = 0.0;
    }
  } else {
    // key_range_size is 0: invalid, use NaN to expose error
    features->overlap_ratio_with_lower = std::numeric_limits<double>::quiet_NaN();
    features->overlap_ratio_with_upper = std::numeric_limits<double>::quiet_NaN();
    if (features->overlap_with_lower > 0 || features->overlap_with_upper > 0) {
      ROCKS_LOG_WARN(info_log,
                     "[ML Features] key_range_size is 0 but overlaps exist: "
                     "overlap_with_lower=%" PRIu64 " overlap_with_upper=%" PRIu64
                     ", file_number=%" PRIu64 ", level=%d",
                     features->overlap_with_lower, features->overlap_with_upper,
                     file_number, level);
    }
  }

  // Calculate key_range_position_in_level and key_range_percentile_in_level
  int position = 0;
  for (const auto* f : level_files) {
    if (f == nullptr) continue;
    // Compare using smallest key
    if (icmp->Compare(f->smallest, file_smallest) < 0) {
      position++;
    }
  }
  features->key_range_position_in_level = static_cast<double>(position);
  if (current_file_count > 1) {
    features->key_range_percentile_in_level = 
        static_cast<double>(position) / static_cast<double>(current_file_count - 1);
  } else {
    features->key_range_percentile_in_level = 0.0;
  }

  // Calculate neighbor distances
  // Store neighbor information as value types (mimicking RocksDB native approach)
  struct NeighborInfo {
    InternalKey smallest;
    InternalKey largest;
    const FileMetaData* ptr;  // Keep pointer for CalculateKeyDistance
  };
  NeighborInfo left_neighbor_info = {InternalKey(), InternalKey(), nullptr};
  NeighborInfo right_neighbor_info = {InternalKey(), InternalKey(), nullptr};
  
  for (const auto* other_file : level_files) {
    // Safety check: report error for null or invalid file pointers
    if (other_file == nullptr) {
      ROCKS_LOG_ERROR(info_log,
                      "[ML Features] Invalid null pointer in level_files "
                      "neighbor calculation at level %d, target_file_number=%" PRIu64,
                      level, file_number);
      continue;
    }
    // Extract fields to local variables (mimicking RocksDB native approach)
    InternalKey other_smallest = other_file->smallest;
    InternalKey other_largest = other_file->largest;
    FileDescriptor other_fd = other_file->fd;  // Copy for logging
    
    if (other_smallest.unset() || other_largest.unset()) {
      ROCKS_LOG_ERROR(info_log,
                      "[ML Features] Invalid file with unset keys in "
                      "level_files neighbor calculation at level %d, "
                      "file_number=%" PRIu64 ", target_file_number=%" PRIu64,
                      level, other_fd.GetNumber(), file_number);
      continue;
    }
    // Use already extracted file_smallest and file_largest instead of accessing file pointer
    if (icmp->Compare(other_largest, file_smallest) < 0) {
      if (left_neighbor_info.ptr == nullptr ||
          icmp->Compare(other_largest, left_neighbor_info.largest) > 0) {
        left_neighbor_info.smallest = other_smallest;
        left_neighbor_info.largest = other_largest;
        left_neighbor_info.ptr = other_file;
      }
    } else if (icmp->Compare(other_smallest, file_largest) > 0) {
      if (right_neighbor_info.ptr == nullptr ||
          icmp->Compare(other_smallest, right_neighbor_info.smallest) < 0) {
        right_neighbor_info.smallest = other_smallest;
        right_neighbor_info.largest = other_largest;
        right_neighbor_info.ptr = other_file;
      }
    }
  }
  
  const FileMetaData* left_neighbor = left_neighbor_info.ptr;
  const FileMetaData* right_neighbor = right_neighbor_info.ptr;

  if (left_neighbor != nullptr) {
    // Calculate distance using temp_file (which contains our file's keys)
    features->left_neighbor_key_distance =
        CalculateKeyDistance(&temp_file, left_neighbor, icmp, info_log);
  } else {
    // No left neighbor, use -1 to indicate no neighbor
    features->left_neighbor_key_distance = -1.0;
  }

  if (right_neighbor != nullptr) {
    // Calculate distance using temp_file (which contains our file's keys)
    features->right_neighbor_key_distance =
        CalculateKeyDistance(&temp_file, right_neighbor, icmp, info_log);
  } else {
    // No right neighbor, use -1 to indicate no neighbor
    features->right_neighbor_key_distance = -1.0;
  }

  // Calculate min_neighbor_key_distance, handling -1 (no neighbor) case
  if (features->left_neighbor_key_distance >= 0 && features->right_neighbor_key_distance >= 0) {
    // Both neighbors exist
    features->min_neighbor_key_distance = std::min(
        features->left_neighbor_key_distance, features->right_neighbor_key_distance);
  } else if (features->left_neighbor_key_distance >= 0) {
    // Only left neighbor exists
    features->min_neighbor_key_distance = features->left_neighbor_key_distance;
  } else if (features->right_neighbor_key_distance >= 0) {
    // Only right neighbor exists
    features->min_neighbor_key_distance = features->right_neighbor_key_distance;
  } else {
    // No neighbors
    features->min_neighbor_key_distance = -1.0;
  }
  
  // Calculate average neighbor distance
  // Formula: avg_neighbor_key_distance = (left_neighbor_key_distance + right_neighbor_key_distance) / 2
  // Semantics:
  //   - Value small (close to 0): files are tightly adjacent → files concentrated → easier to be deleted by extension
  //   - Value large: files are far from neighbors → files isolated → harder to be deleted by extension
  //   - Value -1: no neighbors (file is first/last in level, or only file in level) → completely isolated → won't be deleted by extension
  if (left_neighbor != nullptr && right_neighbor != nullptr) {
    features->avg_neighbor_key_distance =
        (features->left_neighbor_key_distance +
         features->right_neighbor_key_distance) /
        2.0;
  } else if (left_neighbor != nullptr) {
    // Only left neighbor exists, use its distance
    features->avg_neighbor_key_distance = features->left_neighbor_key_distance;
  } else if (right_neighbor != nullptr) {
    // Only right neighbor exists, use its distance
    features->avg_neighbor_key_distance = features->right_neighbor_key_distance;
  } else {
    // No neighbors (file is first/last in level, or only file in level)
    // Set to -1 to indicate complete isolation
    features->avg_neighbor_key_distance = -1.0;
  }

  // ========== Competition Features (56-59) ==========
  
  // Count files with better/worse scores
  // When all scores are 0, use key position as tiebreaker
  bool all_active_scores_zero = (features->first_active_score == 0.0);
  for (const auto& info : file_scores) {
    if (info.file != &temp_file) {
      if (info.active_score < features->first_active_score) {
        features->better_score_files_count++;
      } else if (info.active_score > features->first_active_score) {
        features->worse_score_files_count++;
      } else if (all_active_scores_zero && info.active_score == 0.0) {
        // When all scores are 0, use key position as tiebreaker
        // Files with smaller smallest key are considered "better"
        if (icmp->Compare(info.file->smallest, file_smallest) < 0) {
          features->better_score_files_count++;
        } else if (icmp->Compare(info.file->smallest, file_smallest) > 0) {
          features->worse_score_files_count++;
        }
      }
    }
  }

  // competition_ratio
  if (current_file_count > 0) {
    features->competition_ratio =
        static_cast<double>(features->better_score_files_count) /
        current_file_count;
  }

  // neighbor_files_count - count files that are close neighbors
  features->neighbor_files_count = 0;
  if (features->left_neighbor_key_distance >= 0.0 &&
      features->left_neighbor_key_distance <= 1.0) {
    features->neighbor_files_count++;
  }
  if (features->right_neighbor_key_distance >= 0.0 &&
      features->right_neighbor_key_distance <= 1.0) {
    features->neighbor_files_count++;
  }

  // ========== Cross Level Features (60-61) ==========
  
  // lower_level_capacity_ratio = lower_level_size / MaxBytesForLevel(lower_level)
  if (level < vstorage->num_levels() - 1) {
    int lower_level = level + 1;
    uint64_t lower_level_size = vstorage->NumLevelBytes(lower_level);
    uint64_t lower_level_max = vstorage->MaxBytesForLevel(lower_level);
    if (lower_level_max > 0) {
      features->lower_level_capacity_ratio = 
          static_cast<double>(lower_level_size) / static_cast<double>(lower_level_max);
    }
  }
  
  // upper_level_capacity_ratio = upper_level_size / MaxBytesForLevel(upper_level)
  if (level > 0) {
    int upper_level = level - 1;
    uint64_t upper_level_size = vstorage->NumLevelBytes(upper_level);
    uint64_t upper_level_max = vstorage->MaxBytesForLevel(upper_level);
    if (upper_level_max > 0) {
      features->upper_level_capacity_ratio = 
          static_cast<double>(upper_level_size) / static_cast<double>(upper_level_max);
    }
  }

  // ========== Composite Scores ==========
  
  // urgency_score = file_count_ratio * (1 / active_rank_normalized)
  if (features->active_rank_normalized > 0) {
    features->urgency_score = features->file_count_ratio / features->active_rank_normalized;
  }

  // health_score = 1 / (1 + level_avg_score)
  features->health_score = 1.0 / (1.0 + features->level_avg_score);
  
  // stability_score = 1 / (1 + overlap_ratio_with_lower + overlap_ratio_with_upper)
  features->stability_score = 1.0 / (1.0 + features->overlap_ratio_with_lower + features->overlap_ratio_with_upper);

  // Calculation complete, version_guard will release the reference in its destructor
  // Mutex will be released when lock goes out of scope
  return true;
}

// Helper function to write double value to JSON, handling -1 (no neighbor)
// -1 indicates no neighbor exists
[[maybe_unused]] static void WriteDoubleOrInf(JSONWriter* jwriter, const char* key, double value) {
  *jwriter << key;
  // -1 indicates no neighbor, write as -1
  *jwriter << value;
}

void WriteMLFeaturesToJSON(const MLFeatures& features, JSONWriter* jwriter) {
  jwriter->StartObject();

  // Rank/Score features (8 features)
  *jwriter << "first_active_rank" << features.first_active_rank
          << "first_passive_rank" << features.first_passive_rank
          << "active_rank_normalized" << features.active_rank_normalized
          << "passive_rank_normalized" << features.passive_rank_normalized
          << "first_active_score" << features.first_active_score
          << "first_passive_score" << features.first_passive_score
          << "level_avg_score" << features.level_avg_score
          << "score_rank_interaction" << features.score_rank_interaction;

  // Key Range features (4 features)
  *jwriter << "key_range" << features.key_range
          << "key_range_start" << static_cast<int64_t>(features.key_range_start)
          << "key_range_end" << static_cast<int64_t>(features.key_range_end)
          << "key_range_size" << static_cast<int64_t>(features.key_range_size)
          << "log10_key_range_size" << features.log10_key_range_size;

  // Level Stats features (9 features)
  *jwriter << "level0_cumulative_compaction_count" << features.level0_cumulative_compaction_count
          << "level1_cumulative_compaction_count" << features.level1_cumulative_compaction_count
          << "level2_cumulative_compaction_count" << features.level2_cumulative_compaction_count
          << "level2_total_size" << static_cast<int64_t>(features.level2_total_size)
          << "level4_current_file_count" << features.level4_current_file_count
          << "level4_cumulative_compaction_count" << features.level4_cumulative_compaction_count
          << "level5_current_file_count" << features.level5_current_file_count
          << "level5_cumulative_compaction_count" << features.level5_cumulative_compaction_count
          << "level5_total_size" << static_cast<int64_t>(features.level5_total_size);

  // Overlap features (6 features)
  *jwriter << "overlap_with_lower" << static_cast<int64_t>(features.overlap_with_lower)
          << "overlap_with_upper" << static_cast<int64_t>(features.overlap_with_upper)
          << "overlap_count_with_lower" << features.overlap_count_with_lower
          << "overlap_count_with_upper" << features.overlap_count_with_upper
          << "overlap_ratio_with_lower" << features.overlap_ratio_with_lower
          << "overlap_ratio_with_upper" << features.overlap_ratio_with_upper;
  
  // Neighbor features (4 features)
  WriteDoubleOrInf(jwriter, "left_neighbor_key_distance", features.left_neighbor_key_distance);
  WriteDoubleOrInf(jwriter, "right_neighbor_key_distance", features.right_neighbor_key_distance);
  WriteDoubleOrInf(jwriter, "min_neighbor_key_distance", features.min_neighbor_key_distance);
  WriteDoubleOrInf(jwriter, "avg_neighbor_key_distance", features.avg_neighbor_key_distance);

  // Competition features (3 features)
  *jwriter << "better_score_files_count" << features.better_score_files_count
          << "worse_score_files_count" << features.worse_score_files_count
          << "competition_ratio" << features.competition_ratio;

  // Other features (1 feature)
  *jwriter << "urgency_score" << features.urgency_score;

  // Creation Level (for logging only)
  *jwriter << "creation_level" << features.creation_level;

  jwriter->EndObject();
}

// Calculate ML features before file write (file_size not available)
// This function reuses the complete logic from CalculateMLFeatures but removes
// file_size related features (file_size, log10_file_size, file_size_ratio, file_density, log10_file_density)
static inline bool SubsetHasAnyInRange(const std::vector<int>* subset, int lo, int hi) {
  if (subset == nullptr || subset->empty()) return true;
  for (int i : *subset) {
    if (i >= lo && i <= hi) return true;
  }
  return false;
}

bool CalculateMLFeaturesBeforeWrite(const InternalKey& smallest,
                                    const InternalKey& largest,
                                    uint64_t num_entries,
                                    uint64_t estimated_file_size,
                                    int level,
                                    ColumnFamilyData* cfd,
                                    MLFeatures* features,
                                    InstrumentedMutex* db_mutex,
                                    const std::vector<int>* subset_indices) {
  Logger* info_log = cfd ? cfd->ioptions().info_log.get() : nullptr;
  const bool need_rank = SubsetHasAnyInRange(subset_indices, 0, 10);
  const bool need_key = SubsetHasAnyInRange(subset_indices, 11, 16);
  const bool need_level = SubsetHasAnyInRange(subset_indices, 17, 49);
  const bool need_overlap = SubsetHasAnyInRange(subset_indices, 50, 55);
  const bool need_neighbor = SubsetHasAnyInRange(subset_indices, 56, 59);
  const bool need_competition = SubsetHasAnyInRange(subset_indices, 60, 63);
  const bool need_other = SubsetHasAnyInRange(subset_indices, 64, 68);
  const bool need_key_effective = need_key || need_overlap;
  const bool need_rank_effective = need_rank || need_competition;
  
  // Validate parameters
  if (cfd == nullptr || features == nullptr) {
    if (info_log) {
      ROCKS_LOG_ERROR(info_log,
                      "[ML Features] Invalid parameters: cfd=%p, features=%p",
                      static_cast<const void*>(cfd),
                      static_cast<const void*>(features));
    }
    return false;
  }

  // Only calculate for kMinOverlappingRatio
  if (cfd->ioptions().compaction_pri != kMinOverlappingRatio) {
    if (info_log) {
      ROCKS_LOG_DEBUG(info_log,
                      "[ML Features] Skipping: compaction_pri=%d != kMinOverlappingRatio "
                      "for level=%d",
                      static_cast<int>(cfd->ioptions().compaction_pri), level);
    }
    return false;
  }

  InternalKey file_smallest = smallest;
  InternalKey file_largest = largest;

  // Level 0: skip ML features (fixed to handle 6). Level 1–5: compute features for prediction.
  if (level == 0) {
    return false;
  }

  // Initialize all features to 0
  *features = MLFeatures{};

  // Set creation_level (this is always available, even before other features are calculated)
  features->creation_level = level;

  // Check if file has valid keys first
  if (file_smallest.unset() || file_largest.unset()) {
    if (info_log) {
      ROCKS_LOG_WARN(info_log,
                     "[ML Features] File has unset keys: level=%d, smallest_unset=%d, largest_unset=%d",
                     level, file_smallest.unset(), file_largest.unset());
    }
    return false;
  }

  // Use mutex to protect entire calculation process
  std::unique_ptr<InstrumentedMutexLock> lock;
  if (db_mutex != nullptr) {
    lock.reset(new InstrumentedMutexLock(db_mutex));
  }
  
  // Get Version and VersionStorageInfo under mutex protection
  Version* version = cfd->current();
  if (version == nullptr) {
    if (info_log) {
      ROCKS_LOG_ERROR(info_log,
                      "[ML Features] current() returned nullptr: level=%d",
                      level);
    }
    return false;
  }
  
  // Increment reference count to prevent Version from being deleted
  version->Ref();
  VersionStorageInfo* vstorage = version->storage_info();
  
  if (vstorage == nullptr) {
    version->Unref();
    if (info_log) {
      ROCKS_LOG_ERROR(info_log,
                      "[ML Features] vstorage is nullptr: level=%d",
                      level);
    }
    return false;
  }
  
  // RAII helper to ensure Version reference is released
  struct VersionRefGuard {
    Version* version_;
    
    VersionRefGuard(Version* v) : version_(v) {}
    
    ~VersionRefGuard() {
      if (version_ != nullptr) {
        version_->Unref();
      }
    }
    
    // Non-copyable
    VersionRefGuard(const VersionRefGuard&) = delete;
    VersionRefGuard& operator=(const VersionRefGuard&) = delete;
  };
  
  VersionRefGuard version_guard(version);
  
  const auto& ioptions = cfd->ioptions();
  const auto& mutable_cf_options = cfd->GetCurrentMutableCFOptions();
  const InternalKeyComparator* icmp = &cfd->internal_comparator();

  if (level < 0 || level >= vstorage->num_levels()) {
    if (info_log) {
      ROCKS_LOG_ERROR(info_log,
                      "[ML Features] Invalid level %d (num_levels=%d)",
                      level, vstorage->num_levels());
    }
    return false;
  }

  // Estimate compensated_file_size
  // Priority: 1) Use estimated_file_size if provided (e.g., target_file_size)
  //           2) Fall back to num_entries * 100 if estimated_file_size is 0
  //           3) Use a reasonable default (16MB) if both are 0
  uint64_t estimated_compensated_file_size;
  if (estimated_file_size > 0) {
    estimated_compensated_file_size = estimated_file_size;
  } else if (num_entries > 0) {
    estimated_compensated_file_size = num_entries * 100;
  } else {
    // Default to 16MB (typical SST file size)
    estimated_compensated_file_size = 16 * 1024 * 1024;
  }
  
  // Validation LOG: Input parameters
  if (info_log) {
    Slice smallest_user_key = ExtractUserKey(file_smallest.Encode());
    Slice largest_user_key = ExtractUserKey(file_largest.Encode());
    ROCKS_LOG_INFO(info_log,
                   "[ML Features Validation] Input: level=%d num_entries=%" PRIu64
                   " estimated_file_size=%" PRIu64 " estimated_compensated_file_size=%" PRIu64
                   " smallest_key_len=%zu largest_key_len=%zu",
                   level, num_entries, estimated_file_size, estimated_compensated_file_size,
                   smallest_user_key.size(), largest_user_key.size());
  }

  // Get current level files (excluding current file if not yet added)
  std::vector<FileMetaData*> level_files;
  try {
    int level_files_count = vstorage->NumLevelFiles(level);
    if (level_files_count > 0) {
      const std::vector<FileMetaData*>& level_files_ref = vstorage->LevelFiles(level);
      const size_t kMaxReasonableFiles = 1000000;
      size_t level_files_size = static_cast<size_t>(level_files_count);
      if (level_files_size > kMaxReasonableFiles) {
        if (info_log) {
          ROCKS_LOG_ERROR(info_log,
                          "[ML Features] Invalid level_files size %zu (too large) at level %d",
                          level_files_size, level);
        }
        level_files_size = 0;
      }
      
      level_files.reserve(level_files_size);
      for (size_t i = 0; i < level_files_size; ++i) {
        const FileMetaData* f = nullptr;
        try {
          f = level_files_ref[i];
        } catch (...) {
          if (info_log) {
            ROCKS_LOG_ERROR(info_log,
                            "[ML Features] Failed to access level_files[%zu] at level %d",
                            i, level);
          }
          continue;
        }
        if (f != nullptr) {
          level_files.push_back(const_cast<FileMetaData*>(f));
        }
      }
    }
  } catch (...) {
    if (info_log) {
      ROCKS_LOG_ERROR(info_log,
                      "[ML Features] Exception getting level_files at level %d",
                      level);
    }
    // Continue with empty level_files
  }
  
  // Create a temporary file list that includes current file for score calculation
  std::vector<FileMetaData*> files_with_current = level_files;
  FileMetaData temp_file;
  try {
    // Create a temporary FileDescriptor (file_number will be set when file is created)
    // Use FileDescriptor constructor: FileDescriptor(number, path_id, file_size)
    FileDescriptor temp_fd(0, 0, estimated_compensated_file_size);
    temp_file.fd = temp_fd;
    temp_file.smallest = file_smallest;
    temp_file.largest = file_largest;
    temp_file.compensated_file_size = estimated_compensated_file_size;
    temp_file.num_entries = num_entries;
    
    // Set oldest_ancester_time to current time for newly created files
    int64_t curr_time = 0;
    Status time_status = ioptions.clock->GetCurrentTime(&curr_time);
    if (time_status.ok()) {
      temp_file.oldest_ancester_time = static_cast<uint64_t>(curr_time);
    } else {
      temp_file.oldest_ancester_time = 0;
    }
  } catch (...) {
    if (info_log) {
      ROCKS_LOG_ERROR(info_log,
                      "[ML Features] Exception creating temp_file at level %d",
                      level);
    }
    return false;
  }
  files_with_current.push_back(&temp_file);
  
  // 预先计算 current_file_count（在条件块外使用）
  int current_file_count = vstorage->NumLevelFiles(level) + 1;
  
  struct FileScoreInfo {
    const FileMetaData* file;
    double active_score;
    double passive_score;
  };
  std::vector<FileScoreInfo> file_scores;

  if (need_rank_effective) {
  // ========== Score and Rank Features (0-10) ==========
  
  // 所有 Level (1-5) 都计算 score/rank 特征
  // Calculate active_score using estimated file size
  double active_score_raw = CalculateFileScoreForMinOverlappingRatio(
      &file_smallest, &file_largest, 0, estimated_compensated_file_size, vstorage, level, icmp, ioptions,
      mutable_cf_options, info_log, db_mutex);
  
  // Apply TTL boost and compensated_file_size normalization
  uint64_t ttl = mutable_cf_options.ttl;
  int64_t curr_time;
  Status status = ioptions.clock->GetCurrentTime(&curr_time);
  if (!status.ok()) {
    ttl = 0;
  }
  
  FileTtlBooster ttl_booster(static_cast<uint64_t>(curr_time), ttl,
                             vstorage->num_non_empty_levels(), level);
  uint64_t ttl_boost_score = (ttl > 0) ? ttl_booster.GetBoostScore(&temp_file) : 1;
  assert(ttl_boost_score > 0);
  assert(estimated_compensated_file_size != 0);
  
  // Calculate score exactly as RocksDB does: overlapping_bytes * 1024U / compensated_file_size / ttl_boost_score
  uint64_t score = static_cast<uint64_t>(active_score_raw) * 1024U /
                   estimated_compensated_file_size /
                   ttl_boost_score;
  features->first_active_score = static_cast<double>(score);
  
  // Validation LOG: Score calculation
  if (info_log) {
    ROCKS_LOG_INFO(info_log,
                   "[ML Features Validation] Score calculation: level=%d"
                   " active_score_raw=%.2f estimated_compensated_file_size=%" PRIu64
                   " ttl_boost_score=%" PRIu64 " first_active_score=%.2f",
                   level, active_score_raw, estimated_compensated_file_size,
                   ttl_boost_score, features->first_active_score);
  }

  // Calculate first_passive_score
  double passive_score_raw = CalculatePassiveScoreForMinOverlappingRatio(
      &file_smallest, &file_largest, 0, estimated_compensated_file_size, vstorage, level, icmp, ioptions,
      mutable_cf_options, info_log, db_mutex);
  
  // Calculate passive score exactly as active score (same TTL boost logic)
  uint64_t passive_ttl = mutable_cf_options.ttl;
  int64_t passive_curr_time;
  Status passive_status = ioptions.clock->GetCurrentTime(&passive_curr_time);
  if (!passive_status.ok()) {
    passive_ttl = 0;
  }
  
  FileTtlBooster passive_ttl_booster(static_cast<uint64_t>(passive_curr_time), passive_ttl,
                                    vstorage->num_non_empty_levels(), level);
  uint64_t passive_ttl_boost_score = (passive_ttl > 0) ? passive_ttl_booster.GetBoostScore(&temp_file) : 1;
  assert(passive_ttl_boost_score > 0);
  assert(estimated_compensated_file_size != 0);
  
  uint64_t passive_score_value = static_cast<uint64_t>(passive_score_raw) * 1024U /
                                  estimated_compensated_file_size /
                                  passive_ttl_boost_score;
  features->first_passive_score = static_cast<double>(passive_score_value);

  // Calculate scores for all files in level to determine rank
  // FileScoreInfo 已在条件块外定义
  
  const size_t max_files_for_rank_calc = 1000;
  const size_t files_to_process = std::min(level_files.size(), max_files_for_rank_calc);
  
  // file_scores 已在条件块外定义
  file_scores.reserve(files_to_process + 1);

  // Calculate scores for existing files
  size_t processed = 0;
  for (const auto* f : level_files) {
    if (processed >= files_to_process) {
      break;
    }
    if (f == nullptr) {
      if (info_log) {
        ROCKS_LOG_ERROR(info_log,
                        "[ML Features] Invalid null pointer in level_files "
                        "rank calculation at level %d",
                        level);
      }
      continue;
    }
    if (f->smallest.unset() || f->largest.unset()) {
      if (info_log) {
        ROCKS_LOG_ERROR(info_log,
                        "[ML Features] Invalid file with unset keys in "
                        "level_files rank calculation at level %d",
                        level);
      }
      continue;
    }
    FileDescriptor f_fd = f->fd;
    InternalKey f_smallest = f->smallest;
    InternalKey f_largest = f->largest;
    uint64_t f_number = f_fd.GetNumber();
    uint64_t f_compensated_size = f->compensated_file_size;
    
    double f_active_score_raw = CalculateFileScoreForMinOverlappingRatio(
        &f_smallest, &f_largest, f_number, f_compensated_size, vstorage, level, icmp, ioptions,
        mutable_cf_options, info_log, db_mutex);
    
    uint64_t f_ttl = mutable_cf_options.ttl;
    int64_t f_curr_time;
    Status f_status = ioptions.clock->GetCurrentTime(&f_curr_time);
    if (!f_status.ok()) {
      f_ttl = 0;
    }
    
    FileTtlBooster f_ttl_booster(static_cast<uint64_t>(f_curr_time), f_ttl,
                                vstorage->num_non_empty_levels(), level);
    uint64_t f_ttl_boost_score = (f_ttl > 0) ? f_ttl_booster.GetBoostScore(const_cast<FileMetaData*>(f)) : 1;
    assert(f_ttl_boost_score > 0);
    assert(f_compensated_size != 0);
    
    uint64_t f_score = static_cast<uint64_t>(f_active_score_raw) * 1024U /
                      f_compensated_size /
                      f_ttl_boost_score;
    double active_score = static_cast<double>(f_score);
    
    double f_passive_score_raw = CalculatePassiveScoreForMinOverlappingRatio(
        &f_smallest, &f_largest, f_number, f_compensated_size, vstorage, level, icmp, ioptions,
        mutable_cf_options, info_log, db_mutex);
    
    uint64_t f_passive_ttl = mutable_cf_options.ttl;
    int64_t f_passive_curr_time;
    Status f_passive_status = ioptions.clock->GetCurrentTime(&f_passive_curr_time);
    if (!f_passive_status.ok()) {
      f_passive_ttl = 0;
    }
    
    FileTtlBooster f_passive_ttl_booster(static_cast<uint64_t>(f_passive_curr_time), f_passive_ttl,
                                         vstorage->num_non_empty_levels(), level);
    uint64_t f_passive_ttl_boost_score = (f_passive_ttl > 0) ? f_passive_ttl_booster.GetBoostScore(const_cast<FileMetaData*>(f)) : 1;
    assert(f_passive_ttl_boost_score > 0);
    assert(f_compensated_size != 0);
    
    uint64_t f_passive_score = static_cast<uint64_t>(f_passive_score_raw) * 1024U /
                              f_compensated_size /
                              f_passive_ttl_boost_score;
    double passive_score = static_cast<double>(f_passive_score);
    file_scores.push_back({f, active_score, passive_score});
    processed++;
  }
  // Add current file
  file_scores.push_back({&temp_file, features->first_active_score, features->first_passive_score});

  // Calculate active_rank by sorting files by score
  // Sort in ascending order of score (lower score = better = lower rank)
  // This matches RocksDB's kMinOverlappingRatio strategy: prioritize files with less overlap
  // First prioritize marked_for_compaction=true files (exactly as RocksDB does)
  std::sort(file_scores.begin(), file_scores.end(),
            [&](const FileScoreInfo& a, const FileScoreInfo& b) {
              // First compare by marked_for_compaction (exactly as RocksDB does)
              if (a.file->marked_for_compaction != b.file->marked_for_compaction) {
                return a.file->marked_for_compaction > b.file->marked_for_compaction;
              }
              // Then compare by score (lower is better for kMinOverlappingRatio)
              if (a.active_score != b.active_score) {
                return a.active_score < b.active_score;
              }
              // If scores are equal, compare by key (smaller is better)
              return icmp->Compare(a.file->smallest, b.file->smallest) < 0;
            });
  
  int active_rank = 1;
  for (size_t i = 0; i < file_scores.size(); i++) {
    if (file_scores[i].file == &temp_file) {
      size_t first_same_score_idx = i;
      // When finding files with the same rank, we must consider both
      // marked_for_compaction and active_score, matching the sort order above
      while (first_same_score_idx > 0) {
        const FileScoreInfo& prev = file_scores[first_same_score_idx - 1];
        const FileScoreInfo& curr = file_scores[i];
        // Two files have the same rank if they have the same marked_for_compaction
        // and the same active_score (and same key, but that's already handled by sort)
        if (prev.file->marked_for_compaction == curr.file->marked_for_compaction &&
            prev.active_score == curr.active_score) {
          first_same_score_idx--;
        } else {
          break;
        }
      }
      active_rank = static_cast<int>(first_same_score_idx + 1);
      break;
    }
  }
  features->first_active_rank = active_rank;

  // Calculate passive_rank
  // Sort in ascending order of score (lower score = better = lower rank)
  // First prioritize marked_for_compaction=true files (exactly as RocksDB does)
  std::sort(file_scores.begin(), file_scores.end(),
            [&](const FileScoreInfo& a, const FileScoreInfo& b) {
              // First compare by marked_for_compaction (exactly as RocksDB does)
              if (a.file->marked_for_compaction != b.file->marked_for_compaction) {
                return a.file->marked_for_compaction > b.file->marked_for_compaction;
              }
              // Then compare by score (lower is better, same as active_rank)
              if (a.passive_score != b.passive_score) {
                return a.passive_score < b.passive_score;
              }
              // If scores are equal, compare by key (smaller is better)
              return icmp->Compare(a.file->smallest, b.file->smallest) < 0;
            });
  
  int passive_rank = 1;
  for (size_t i = 0; i < file_scores.size(); i++) {
    if (file_scores[i].file == &temp_file) {
      size_t first_same_score_idx = i;
      // When finding files with the same rank, we must consider both
      // marked_for_compaction and passive_score, matching the sort order above
      while (first_same_score_idx > 0) {
        const FileScoreInfo& prev = file_scores[first_same_score_idx - 1];
        const FileScoreInfo& curr = file_scores[i];
        // Two files have the same rank if they have the same marked_for_compaction
        // and the same passive_score (and same key, but that's already handled by sort)
        if (prev.file->marked_for_compaction == curr.file->marked_for_compaction &&
            prev.passive_score == curr.passive_score) {
          first_same_score_idx--;
        } else {
          break;
        }
      }
      passive_rank = static_cast<int>(first_same_score_idx + 1);
      break;
    }
  }
  features->first_passive_rank = passive_rank;

  // Calculate normalized ranks
  // current_file_count 已在条件块外定义
  if (current_file_count > 0) {
    features->active_rank_normalized =
        static_cast<double>(features->first_active_rank) / current_file_count;
    features->passive_rank_normalized =
        static_cast<double>(features->first_passive_rank) / current_file_count;
  }
  
  // Validation LOG: Rank calculation
  if (info_log) {
    ROCKS_LOG_INFO(info_log,
                   "[ML Features Validation] Rank calculation: level=%d"
                   " current_file_count=%d active_rank=%d passive_rank=%d"
                   " active_rank_normalized=%.6f passive_rank_normalized=%.6f",
                   level, current_file_count,
                   features->first_active_rank, features->first_passive_rank,
                   features->active_rank_normalized, features->passive_rank_normalized);
    
    // Validate rank bounds
    if (features->first_active_rank < 1 || features->first_active_rank > current_file_count) {
      ROCKS_LOG_WARN(info_log,
                     "[ML Features Validation] Invalid active_rank=%d (expected 1-%d) at level=%d",
                     features->first_active_rank, current_file_count, level);
    }
    if (features->first_passive_rank < 1 || features->first_passive_rank > current_file_count) {
      ROCKS_LOG_WARN(info_log,
                     "[ML Features Validation] Invalid passive_rank=%d (expected 1-%d) at level=%d",
                     features->first_passive_rank, current_file_count, level);
    }
  }

  // Calculate interaction and difference features
  features->score_rank_interaction =
      features->first_active_score * features->first_active_rank;
  features->rank_difference = features->first_active_rank - features->first_passive_rank;
  features->score_difference = features->first_active_score - features->first_passive_score;

  // Calculate level_avg_score (uses file_scores, must stay inside need_rank_effective scope)
  double sum_scores = 0.0;
  for (const auto& info : file_scores) {
    sum_scores += info.active_score;
  }
  if (file_scores.size() > 0) {
    features->level_avg_score = sum_scores / file_scores.size();
  }

  }  // need_rank_effective

  // ========== Level State Features (file_count_ratio, level* 17-49) ==========
  
  // Get cumulative_file_count from InternalStats
  InternalStats* internal_stats = cfd->internal_stats();
  uint64_t current_level_cumulative_file_count = current_file_count;
  if (internal_stats != nullptr) {
    // TODO: GetCumulativeFileCount method not yet implemented in InternalStats
    // current_level_cumulative_file_count = internal_stats->GetCumulativeFileCount(level);
    if (current_level_cumulative_file_count < static_cast<uint64_t>(current_file_count)) {
      if (info_log) {
        ROCKS_LOG_DEBUG(info_log,
                       "[ML Features] cumulative_file_count (%" PRIu64 ") < current_file_count (%d) for level %d. "
                       "Using current_file_count as minimum.",
                       current_level_cumulative_file_count, current_file_count, level);
      }
      current_level_cumulative_file_count = static_cast<uint64_t>(current_file_count);
    }
  }

  // Calculate file_count_ratio
  if (level == 0) {
    int compaction_trigger = mutable_cf_options.level0_file_num_compaction_trigger;
    if (compaction_trigger > 0) {
      features->file_count_ratio =
          static_cast<double>(current_file_count) / compaction_trigger;
    }
  } else {
    // For level 1+, trigger is based on MaxBytesForLevel
    uint64_t level_size = 0;
    for (const auto* f : level_files) {
      if (f == nullptr) {
        continue;
      }
      FileDescriptor f_fd = f->fd;
      level_size += f_fd.GetFileSize();
    }
    level_size += estimated_compensated_file_size; // Add estimated size for current file
    if (level_size >= vstorage->MaxBytesForLevel(level)) {
      features->file_count_ratio = static_cast<double>(current_file_count);
    } else {
      features->file_count_ratio = 0.0;
    }
  }

  if (need_level) {
  // Get file counts and total sizes for all levels (0-6)
  for (int l = 0; l < 7 && l < vstorage->num_levels(); l++) {
    int count = 0;
    uint64_t total_size = 0;
    try {
      count = vstorage->NumLevelFiles(l);
      
      // Calculate total size for this level
      if (count > 0) {
        try {
          total_size = vstorage->NumLevelBytes(l);
        } catch (...) {
          if (info_log) {
            ROCKS_LOG_DEBUG(info_log,
                            "[ML Features] Exception getting NumLevelBytes at level %d",
                            l);
          }
          total_size = 0;
        }
      }
    } catch (...) {
      if (info_log) {
        ROCKS_LOG_ERROR(info_log,
                        "[ML Features] Exception getting NumLevelFiles at level %d",
                        l);
      }
    }
    // Consistency: 0 files must imply 0 bytes
    if (count == 0) {
      total_size = 0;
    }
    switch (l) {
      case 0:
        features->level0_current_file_count = count;
        break;
      case 1:
        features->level1_current_file_count = count;
        features->level1_total_size = total_size;
        break;
      case 2:
        features->level2_current_file_count = count;
        features->level2_total_size = total_size;
        break;
      case 3:
        features->level3_current_file_count = count;
        features->level3_total_size = total_size;
        break;
      case 4:
        features->level4_current_file_count = count;
        features->level4_total_size = total_size;
        break;
      case 5:
        features->level5_current_file_count = count;
        features->level5_total_size = total_size;
        break;
      case 6:
        features->level6_current_file_count = count;
        features->level6_total_size = total_size;
        break;
    }
  }

  // Get cumulative compaction count and file count from RocksDB InternalStats
  {
    const std::vector<InternalStats::CompactionStats>* comp_stats_ptr =
        (internal_stats != nullptr) ? &internal_stats->TEST_GetCompactionStats()
                                   : nullptr;
    for (int l = 0; l < 7; l++) {
      uint64_t compaction_count = 0;
      uint64_t cumulative_file_count = 0;
      uint64_t cumulative_trivial_move_count = 0;
      if (comp_stats_ptr != nullptr &&
          l < static_cast<int>(comp_stats_ptr->size())) {
        compaction_count = static_cast<uint64_t>((*comp_stats_ptr)[l].count);
        cumulative_file_count =
            static_cast<uint64_t>((*comp_stats_ptr)[l].num_output_files);
        cumulative_trivial_move_count =
            static_cast<uint64_t>((*comp_stats_ptr)[l].num_trivial_move_files);
      }
      switch (l) {
        case 0:
          features->level0_cumulative_compaction_count = compaction_count;
          features->level0_cumulative_file_count = cumulative_file_count;
          break;
        case 1:
          features->level1_cumulative_compaction_count = compaction_count;
          features->level1_cumulative_file_count = cumulative_file_count;
          features->level1_cumulative_trivial_move_count = cumulative_trivial_move_count;
          break;
        case 2:
          features->level2_cumulative_compaction_count = compaction_count;
          features->level2_cumulative_file_count = cumulative_file_count;
          features->level2_cumulative_trivial_move_count = cumulative_trivial_move_count;
          break;
        case 3:
          features->level3_cumulative_compaction_count = compaction_count;
          features->level3_cumulative_file_count = cumulative_file_count;
          features->level3_cumulative_trivial_move_count = cumulative_trivial_move_count;
          break;
        case 4:
          features->level4_cumulative_compaction_count = compaction_count;
          features->level4_cumulative_file_count = cumulative_file_count;
          features->level4_cumulative_trivial_move_count = cumulative_trivial_move_count;
          break;
        case 5:
          features->level5_cumulative_compaction_count = compaction_count;
          features->level5_cumulative_file_count = cumulative_file_count;
          features->level5_cumulative_trivial_move_count = cumulative_trivial_move_count;
          break;
        case 6:
          features->level6_cumulative_compaction_count = compaction_count;
          features->level6_cumulative_file_count = cumulative_file_count;
          features->level6_cumulative_trivial_move_count = cumulative_trivial_move_count;
          break;
      }
    }
  }
  }  // need_level

  // ========== Key Space Features (11-16 key, 50-55 overlap) ==========
  
  if (need_key_effective) {
  // key_range: Format key range in hex format
  auto SliceToHex = [](const Slice& s) -> std::string {
    std::ostringstream oss;
    oss << std::hex << std::uppercase << std::setfill('0');
    for (size_t i = 0; i < s.size(); ++i) {
      oss << std::setw(2) << static_cast<unsigned int>(static_cast<unsigned char>(s[i]));
    }
    return oss.str();
  };
  
  if (!file_smallest.unset() && !file_largest.unset()) {
    Slice smallest_user_key = file_smallest.user_key();
    Slice largest_user_key = file_largest.user_key();
    std::string smallest_hex = SliceToHex(smallest_user_key);
    std::string largest_hex = SliceToHex(largest_user_key);
    features->key_range = smallest_hex + " .. " + largest_hex;
    
    // If hex string is longer than 16 chars (64 bits), take only the last 16 chars (LSB)
    std::string smallest_hex_truncated = smallest_hex;
    std::string largest_hex_truncated = largest_hex;
    if (smallest_hex.size() > 16) {
      smallest_hex_truncated = smallest_hex.substr(smallest_hex.size() - 16);
    }
    if (largest_hex.size() > 16) {
      largest_hex_truncated = largest_hex.substr(largest_hex.size() - 16);
    }
    try {
      uint64_t smallest_numeric = std::stoull(smallest_hex_truncated, nullptr, 16);
      uint64_t largest_numeric = std::stoull(largest_hex_truncated, nullptr, 16);
      features->key_range_start = smallest_numeric;
      features->key_range_end = largest_numeric;
      if (largest_numeric >= smallest_numeric) {
        features->key_range_size = largest_numeric - smallest_numeric + 1;
      } else {
        features->key_range_size = 0;  // Invalid: largest < smallest
      }
    } catch (...) {
      features->key_range_start = KeyToNumeric(file_smallest);
      features->key_range_end = KeyToNumeric(file_largest);
      features->key_range_size = 0;  // Invalid: hex parse failed
    }
  } else {
    features->key_range = "N/A";
    features->key_range_start = KeyToNumeric(file_smallest);
    features->key_range_end = KeyToNumeric(file_largest);
    features->key_range_size = 0;  // Invalid: keys unset
  }
  features->log10_key_range_size =
      features->key_range_size > 0
          ? std::log10(static_cast<double>(features->key_range_size))
          : std::numeric_limits<double>::quiet_NaN();
  
  // Validation LOG: Key range
  if (info_log) {
    ROCKS_LOG_INFO(info_log,
                   "[ML Features Validation] Key range: level=%d"
                   " key_range=\"%s\" key_range_size=%" PRIu64
                   " log10_key_range_size=%.6f num_entries=%" PRIu64,
                   level, features->key_range.c_str(), features->key_range_size,
                   features->log10_key_range_size, num_entries);
    
    // Validate key range consistency
    if (features->key_range_size == 0 && num_entries > 0) {
      ROCKS_LOG_WARN(info_log,
                     "[ML Features Validation] key_range_size=0 but num_entries=%" PRIu64
                     " at level=%d (key range calculation may have failed)",
                     num_entries, level);
    }
  }
  }  // need_key_effective

  if (need_overlap) {
  // Calculate overlap_with_lower
  if (level < vstorage->num_levels() - 1) {
    std::vector<FileDescriptor> lower_level_file_descriptors;
    try {
      int lower_level = level + 1;
      if (lower_level >= 0 && lower_level < vstorage->num_levels()) {
        std::vector<FileMetaData*> lower_level_files;
        lower_level_files.reserve(100);
        vstorage->GetOverlappingInputs(lower_level, &file_smallest, &file_largest,
                                      &lower_level_files);
        
        lower_level_file_descriptors.reserve(lower_level_files.size());
        for (const FileMetaData* lower_file : lower_level_files) {
          if (lower_file == nullptr) {
            continue;
          }
          try {
            FileDescriptor fd_copy;
            fd_copy.packed_number_and_path_id = lower_file->fd.packed_number_and_path_id;
            fd_copy.file_size = lower_file->fd.file_size;
            fd_copy.smallest_seqno = lower_file->fd.smallest_seqno;
            fd_copy.largest_seqno = lower_file->fd.largest_seqno;
            fd_copy.table_reader = lower_file->fd.table_reader;
            lower_level_file_descriptors.push_back(fd_copy);
          } catch (...) {
            if (info_log) {
              ROCKS_LOG_ERROR(info_log,
                              "[ML Features] Exception copying FileDescriptor in "
                              "lower_level_files at level %d",
                              level + 1);
            }
            continue;
          }
        }
      }
    } catch (...) {
      if (info_log) {
        ROCKS_LOG_ERROR(info_log,
                        "[ML Features] Exception calling GetOverlappingInputs for lower level at level %d",
                        level + 1);
      }
    }
    
    for (const FileDescriptor& lower_fd : lower_level_file_descriptors) {
      try {
        features->overlap_with_lower += lower_fd.GetFileSize();
        features->overlap_count_with_lower++;
      } catch (...) {
        if (info_log) {
          ROCKS_LOG_ERROR(info_log,
                          "[ML Features] Exception getting file size from copied "
                          "FileDescriptor at level %d",
                          level + 1);
        }
        continue;
      }
    }
  }

  // Calculate overlap_with_upper
  if (level > 0) {
    std::vector<FileDescriptor> upper_level_file_descriptors;
    try {
      int upper_level = level - 1;
      if (upper_level >= 0 && upper_level < vstorage->num_levels()) {
        std::vector<FileMetaData*> upper_level_files;
        upper_level_files.reserve(100);
        vstorage->GetOverlappingInputs(upper_level, &file_smallest, &file_largest,
                                       &upper_level_files);
        
        upper_level_file_descriptors.reserve(upper_level_files.size());
        for (const FileMetaData* upper_file : upper_level_files) {
          if (upper_file == nullptr) {
            continue;
          }
          try {
            FileDescriptor fd_copy;
            fd_copy.packed_number_and_path_id = upper_file->fd.packed_number_and_path_id;
            fd_copy.file_size = upper_file->fd.file_size;
            fd_copy.smallest_seqno = upper_file->fd.smallest_seqno;
            fd_copy.largest_seqno = upper_file->fd.largest_seqno;
            fd_copy.table_reader = upper_file->fd.table_reader;
            upper_level_file_descriptors.push_back(fd_copy);
          } catch (...) {
            if (info_log) {
              ROCKS_LOG_ERROR(info_log,
                              "[ML Features] Exception copying FileDescriptor in "
                              "upper_level_files at level %d",
                              level - 1);
            }
            continue;
          }
        }
      }
    } catch (...) {
      if (info_log) {
        ROCKS_LOG_ERROR(info_log,
                        "[ML Features] Exception calling GetOverlappingInputs for upper level at level %d",
                        level - 1);
      }
    }
    
    for (const FileDescriptor& upper_fd : upper_level_file_descriptors) {
      try {
        features->overlap_with_upper += upper_fd.GetFileSize();
        features->overlap_count_with_upper++;
      } catch (...) {
        if (info_log) {
          ROCKS_LOG_ERROR(info_log,
                          "[ML Features] Exception getting file size from copied "
                          "FileDescriptor at level %d",
                          level - 1);
        }
        continue;
      }
    }
  }

  // Note: file_size, log10_file_size, file_size_ratio, file_density, log10_file_density removed
  // These features are not collected before file write

  // Calculate overlap_ratio_with_lower and overlap_ratio_with_upper
  if (features->key_range_size > 0) {
    features->overlap_ratio_with_lower =
        static_cast<double>(features->overlap_with_lower) / features->key_range_size;
    features->overlap_ratio_with_upper =
        static_cast<double>(features->overlap_with_upper) / features->key_range_size;
  } else {
    features->overlap_ratio_with_lower = std::numeric_limits<double>::quiet_NaN();
    features->overlap_ratio_with_upper = std::numeric_limits<double>::quiet_NaN();
  }
  
  // Validation LOG: Overlaps
  if (info_log) {
    ROCKS_LOG_INFO(info_log,
                   "[ML Features Validation] Overlaps: level=%d"
                   " overlap_with_lower=%" PRIu64 " overlap_count_with_lower=%" PRIu64
                   " overlap_ratio_with_lower=%.6f"
                   " overlap_with_upper=%" PRIu64 " overlap_count_with_upper=%" PRIu64
                   " overlap_ratio_with_upper=%.6f",
                   level,
                   features->overlap_with_lower, features->overlap_count_with_lower,
                   features->overlap_ratio_with_lower,
                   features->overlap_with_upper, features->overlap_count_with_upper,
                   features->overlap_ratio_with_upper);
    
    // Validate overlap ratios
    if (features->overlap_ratio_with_lower > 1.0 || features->overlap_ratio_with_upper > 1.0) {
      ROCKS_LOG_WARN(info_log,
                     "[ML Features Validation] Overlap ratio > 1.0 at level=%d:"
                     " overlap_ratio_with_lower=%.6f overlap_ratio_with_upper=%.6f"
                     " (overlap may exceed key_range_size)",
                     level, features->overlap_ratio_with_lower, features->overlap_ratio_with_upper);
    }
  }
  }  // need_overlap

  if (need_key_effective) {
  // Calculate key_range_position_in_level
  int position = 0;
  for (const auto* f : level_files) {
    if (f == nullptr) {
      continue;
    }
    InternalKey f_smallest_key = f->smallest;
    
    if (f_smallest_key.unset()) {
      continue;
    }
    if (icmp->Compare(f_smallest_key, file_smallest) < 0) {
      position++;
    }
  }
  features->key_range_position_in_level = static_cast<double>(position);
  if (current_file_count > 1) {
    features->key_range_percentile_in_level =
        static_cast<double>(position) / static_cast<double>(current_file_count - 1);
  } else {
    features->key_range_percentile_in_level = 0.0;
  }
  }  // need_key_effective

  if (need_neighbor) {
  // Calculate neighbor distances
  struct NeighborInfo {
    InternalKey smallest;
    InternalKey largest;
    const FileMetaData* ptr;
  };
  NeighborInfo left_neighbor_info = {InternalKey(), InternalKey(), nullptr};
  NeighborInfo right_neighbor_info = {InternalKey(), InternalKey(), nullptr};
  
  for (const auto* other_file : level_files) {
    if (other_file == nullptr) {
      continue;
    }
    InternalKey other_smallest = other_file->smallest;
    InternalKey other_largest = other_file->largest;
    
    if (other_smallest.unset() || other_largest.unset()) {
      continue;
    }
    if (icmp->Compare(other_largest, file_smallest) < 0) {
      if (left_neighbor_info.ptr == nullptr ||
          icmp->Compare(other_largest, left_neighbor_info.largest) > 0) {
        left_neighbor_info.smallest = other_smallest;
        left_neighbor_info.largest = other_largest;
        left_neighbor_info.ptr = other_file;
      }
    } else if (icmp->Compare(other_smallest, file_largest) > 0) {
      if (right_neighbor_info.ptr == nullptr ||
          icmp->Compare(other_smallest, right_neighbor_info.smallest) < 0) {
        right_neighbor_info.smallest = other_smallest;
        right_neighbor_info.largest = other_largest;
        right_neighbor_info.ptr = other_file;
      }
    }
  }
  
  const FileMetaData* left_neighbor = left_neighbor_info.ptr;
  const FileMetaData* right_neighbor = right_neighbor_info.ptr;

  if (left_neighbor != nullptr) {
    features->left_neighbor_key_distance =
        CalculateKeyDistance(&temp_file, left_neighbor, icmp, info_log);
  } else {
    features->left_neighbor_key_distance = -1.0;
  }

  if (right_neighbor != nullptr) {
    features->right_neighbor_key_distance =
        CalculateKeyDistance(&temp_file, right_neighbor, icmp, info_log);
  } else {
    features->right_neighbor_key_distance = -1.0;
  }

  // Calculate min_neighbor_key_distance
  if (features->left_neighbor_key_distance >= 0 && features->right_neighbor_key_distance >= 0) {
    features->min_neighbor_key_distance = std::min(
        features->left_neighbor_key_distance, features->right_neighbor_key_distance);
  } else if (features->left_neighbor_key_distance >= 0) {
    features->min_neighbor_key_distance = features->left_neighbor_key_distance;
  } else if (features->right_neighbor_key_distance >= 0) {
    features->min_neighbor_key_distance = features->right_neighbor_key_distance;
  } else {
    features->min_neighbor_key_distance = -1.0;
  }
  
  // Calculate average neighbor distance
  if (left_neighbor != nullptr && right_neighbor != nullptr) {
    features->avg_neighbor_key_distance =
        (features->left_neighbor_key_distance +
         features->right_neighbor_key_distance) /
        2.0;
  } else if (left_neighbor != nullptr) {
    features->avg_neighbor_key_distance = features->left_neighbor_key_distance;
  } else if (right_neighbor != nullptr) {
    features->avg_neighbor_key_distance = features->right_neighbor_key_distance;
  } else {
    features->avg_neighbor_key_distance = -1.0;
  }
  }  // need_neighbor

  if (need_competition) {
  // ========== Competition Features (60-63) ==========
  
  // Count files with better/worse scores
  bool all_active_scores_zero = (features->first_active_score == 0.0);
  for (const auto& info : file_scores) {
    if (info.file != &temp_file) {
      if (info.active_score < features->first_active_score) {
        features->better_score_files_count++;
      } else if (info.active_score > features->first_active_score) {
        features->worse_score_files_count++;
      } else if (all_active_scores_zero && info.active_score == 0.0) {
        if (icmp->Compare(info.file->smallest, file_smallest) < 0) {
          features->better_score_files_count++;
        } else if (icmp->Compare(info.file->smallest, file_smallest) > 0) {
          features->worse_score_files_count++;
        }
      }
    }
  }

  // competition_ratio
  if (current_file_count > 0) {
    features->competition_ratio =
        static_cast<double>(features->better_score_files_count) /
        current_file_count;
  }

  // neighbor_files_count - count files that are close neighbors
  features->neighbor_files_count = 0;
  if (features->left_neighbor_key_distance >= 0.0 &&
      features->left_neighbor_key_distance <= 1.0) {
    features->neighbor_files_count++;
  }
  if (features->right_neighbor_key_distance >= 0.0 &&
      features->right_neighbor_key_distance <= 1.0) {
    features->neighbor_files_count++;
  }
  }  // need_competition

  if (need_other) {
  // ========== Cross Level Features (64-68) ==========
  
  // lower_level_capacity_ratio = lower_level_size / MaxBytesForLevel(lower_level)
  if (level < vstorage->num_levels() - 1) {
    int lower_level = level + 1;
    uint64_t lower_level_size = vstorage->NumLevelBytes(lower_level);
    uint64_t lower_level_max = vstorage->MaxBytesForLevel(lower_level);
    if (lower_level_max > 0) {
      features->lower_level_capacity_ratio = 
          static_cast<double>(lower_level_size) / static_cast<double>(lower_level_max);
    }
  }
  
  // upper_level_capacity_ratio = upper_level_size / MaxBytesForLevel(upper_level)
  if (level > 0) {
    int upper_level = level - 1;
    uint64_t upper_level_size = vstorage->NumLevelBytes(upper_level);
    uint64_t upper_level_max = vstorage->MaxBytesForLevel(upper_level);
    if (upper_level_max > 0) {
      features->upper_level_capacity_ratio = 
          static_cast<double>(upper_level_size) / static_cast<double>(upper_level_max);
    }
  }

  // ========== Composite Scores ==========
  
  // urgency_score = file_count_ratio * (1 / active_rank_normalized)
  if (features->active_rank_normalized > 0) {
    features->urgency_score = features->file_count_ratio / features->active_rank_normalized;
  }

  // health_score = 1 / (1 + level_avg_score)
  features->health_score = 1.0 / (1.0 + features->level_avg_score);
  
  // stability_score = 1 / (1 + overlap_ratio_with_lower + overlap_ratio_with_upper)
  features->stability_score = 1.0 / (1.0 + features->overlap_ratio_with_lower + features->overlap_ratio_with_upper);
  
  // Validation LOG: Final summary (simplified for 35-feature model)
  if (info_log) {
    ROCKS_LOG_INFO(info_log,
                   "[ML Features Validation] Final summary: level=%d"
                   " first_active_score=%.2f first_passive_score=%.2f"
                   " active_rank=%d passive_rank=%d"
                   " level_avg_score=%.2f urgency_score=%.6f"
                   " key_range_size=%" PRIu64 " overlap_with_lower=%" PRIu64
                   " overlap_with_upper=%" PRIu64,
                   level,
                   features->first_active_score, features->first_passive_score,
                   features->first_active_rank, features->first_passive_rank,
                   features->level_avg_score, features->urgency_score,
                   features->key_range_size, features->overlap_with_lower,
                   features->overlap_with_upper);
    
    // Validate score ranges
    if (features->first_active_score < 0.0) {
      ROCKS_LOG_WARN(info_log,
                     "[ML Features Validation] Negative first_active_score=%.2f at level=%d",
                     features->first_active_score, level);
    }
    if (features->first_passive_score < 0.0) {
      ROCKS_LOG_WARN(info_log,
                     "[ML Features Validation] Negative first_passive_score=%.2f at level=%d",
                     features->first_passive_score, level);
    }
    // REMOVED: health_score validation - not needed for reduced feature set
  }
  }  // need_other

  // Calculation complete, version_guard will release the reference in its destructor
  return true;
}

// Log ML features before file write (style consistent with RocksDB logging)
// LogMLFeaturesBeforeWrite is not used, keeping for potential future use
// DISABLED: This function references removed fields, commenting out entire function
namespace {
[[maybe_unused]] void LogMLFeaturesBeforeWrite(Logger* info_log, const std::string& cf_name,
                              uint64_t file_number, int level,
                              const MLFeatures& features) {
  // Function disabled due to reduced feature set
  (void)info_log;
  (void)cf_name;
  (void)file_number;
  (void)level;
  (void)features;
}  // LogMLFeaturesBeforeWrite function - disabled due to reduced feature set
}  // anonymous namespace

#ifdef ROCKSDB_ML_PREDICT_PYTHON
// These functions are in ROCKSDB_NAMESPACE, not anonymous namespace
// Predict file lifetime from ML features
// Returns predicted lifetime in seconds (0 if prediction fails)
// Uses Python ML prediction interface
// file_number: file number for logging (0 if not available)
[[maybe_unused]] static double PredictFileLifetime(const MLFeatures& features, uint64_t file_number,
                           Logger* info_log) {
  // Initialize level-specific predictors once (thread-safe)
  static std::once_flag g_predictor_init_flag;
  std::call_once(g_predictor_init_flag, []() {
    ROCKSDB_NAMESPACE::InitializeMLPredictorByLevel();
  });
  
  // Get the level for this file
  int level = features.creation_level;
  
  // Map level to predictor index:
  // Level 1 -> use Level 1 model
  // Level 2 -> use Level 2 model
  // Level 3 -> use Level 3 model
  // Level 4+ -> use Level 3 model
  int level_for_predict = level;
  if (level_for_predict < 1) {
    return 0.0;
  }
  if (level_for_predict > 6) {
    level_for_predict = 6;
  }
  
  // Check if level is valid
  if (level < 1) {
    return 0.0;  // Use fallback (level-based hint)
  }
  
  // Convert MLFeatures to array (35 features, reduced from 69) in the exact order used for training
  // Order must match training data (from train_models_35_features.py)
  // Build 35 features array matching the reduced feature set
  double features_array[35] = {
    static_cast<double>(features.first_active_rank),  // 0
    static_cast<double>(features.first_passive_rank),  // 1
    features.active_rank_normalized,  // 2
    features.passive_rank_normalized,  // 3
    features.first_active_score,  // 4
    features.first_passive_score,  // 5
    features.level_avg_score,  // 6
    features.score_rank_interaction,  // 7
    static_cast<double>(features.key_range_start),  // 8
    static_cast<double>(features.key_range_end),  // 9
    static_cast<double>(features.key_range_size),  // 10
    features.log10_key_range_size,  // 11
    static_cast<double>(features.level0_cumulative_compaction_count),  // 12
    static_cast<double>(features.level1_cumulative_compaction_count),  // 13
    static_cast<double>(features.level2_cumulative_compaction_count),  // 14
    static_cast<double>(features.level2_total_size),  // 15
    static_cast<double>(features.level4_current_file_count),  // 16
    static_cast<double>(features.level4_cumulative_compaction_count),  // 17
    static_cast<double>(features.level5_current_file_count),  // 18
    static_cast<double>(features.level5_cumulative_compaction_count),  // 19
    static_cast<double>(features.level5_total_size),  // 20
    static_cast<double>(features.overlap_with_lower),  // 21
    static_cast<double>(features.overlap_with_upper),  // 22
    static_cast<double>(features.overlap_count_with_lower),  // 23
    static_cast<double>(features.overlap_count_with_upper),  // 24
    features.overlap_ratio_with_lower,  // 25
    features.overlap_ratio_with_upper,  // 26
    features.left_neighbor_key_distance,  // 27
    features.right_neighbor_key_distance,  // 28
    features.min_neighbor_key_distance,  // 29
    features.avg_neighbor_key_distance,  // 30
    static_cast<double>(features.better_score_files_count),  // 31
    static_cast<double>(features.worse_score_files_count),  // 32
    features.competition_ratio,  // 33
    features.urgency_score,  // 34
  };
  
  // Validate features: NaN, Inf values indicate a problem
  // Log errors but continue to let the problem expose itself
  for (size_t i = 0; i < 35; i++) {
    if (std::isnan(features_array[i])) {
      if (info_log != nullptr) {
        ROCKS_LOG_ERROR(info_log, "[ML Features] ERROR: NaN detected in feature[%zu] at file_number=%" PRIu64 ", level=%d - WILL PASS TO MODEL", 
                        i, file_number, level);
      }
      fprintf(stderr, "[ML Features] ERROR: NaN detected in feature[%zu] at file_number=%" PRIu64 ", level=%d - WILL PASS TO MODEL\n", 
              i, file_number, level);
      // Continue - let the problem expose itself
    }
    if (std::isinf(features_array[i])) {
      if (info_log != nullptr) {
        ROCKS_LOG_ERROR(info_log, "[ML Features] ERROR: Inf detected in feature[%zu] at file_number=%" PRIu64 ", level=%d - WILL PASS TO MODEL", 
                        i, file_number, level);
      }
      fprintf(stderr, "[ML Features] ERROR: Inf detected in feature[%zu] at file_number=%" PRIu64 ", level=%d - WILL PASS TO MODEL\n", 
              i, file_number, level);
      // Continue - let the problem expose itself
    }
  }
  
  // Print ML features in the exact order they will be passed to Python model
  // This matches the order in features_level{1,2,3}.txt
  // Format: {"ML_FEATURES_BEFORE_PYTHON": {"file_number": <num>, "level": <level>, "features": {...}}}
  // Note: These are the values BEFORE scaler standardization (raw features_array values)
  // This is printed right before passing to Python, so the order and values match exactly
  // Print for ALL files (no limit)
  if (info_log != nullptr) {
    // Feature names in the exact order of features_array (matching features_level{1,2,3}.txt)
    static const char* feature_names[32] = {
        "active_rank_normalized",           // 1
        "file_count_ratio",                 // 2
        "first_active_score",               // 3
        "first_passive_score",              // 4
        "key_range_end",                    // 5 (log10 transformed, matching training)
        "key_range_start",                  // 6 (log10 transformed, matching training)
        "level0_cumulative_compaction_count",  // 7
        "level0_current_file_count",        // 8
        "level1_cumulative_compaction_count",  // 9
        "level1_cumulative_trivial_move_count",  // 10
        "level1_current_file_count",        // 11
        "level1_total_size",                 // 12
        "level2_cumulative_compaction_count",  // 13
        "level2_cumulative_trivial_move_count",  // 14
        "level2_current_file_count",        // 15
        "level2_total_size",                 // 16
        "level3_cumulative_compaction_count",  // 17
        "level3_cumulative_trivial_move_count",  // 18
        "level3_current_file_count",        // 19
        "level3_total_size",                 // 20
        "level4_cumulative_compaction_count",  // 21
        "level4_cumulative_trivial_move_count",  // 22
        "level4_current_file_count",        // 23
        "level4_total_size",                 // 24
        "level5_cumulative_compaction_count",  // 25
        "level5_cumulative_trivial_move_count",  // 26
        "level5_current_file_count",        // 27
        "level5_total_size",                 // 28
        "level_avg_score",                  // 29
        "log10_key_range_size",             // 30
        "overlap_with_lower",                // 31
        "passive_rank_normalized",           // 32
    };

    std::ostringstream oss;
    oss.setf(std::ios::fixed);
    oss << std::setprecision(6);
    oss << "{\"ML_FEATURES_BEFORE_PYTHON\": {";
    if (file_number > 0) {
      oss << "\"file_number\": " << file_number << ", ";
    }
    oss << "\"level\": " << level << ", \"features\": {";
    for (size_t i = 0; i < 32; i++) {
      oss << "\"" << feature_names[i] << "\": " << features_array[i];
      if (i < 31) {
        oss << ", ";
      }
    }
    oss << "}}}";
    ROCKS_LOG_INFO(info_log, "%s", oss.str().c_str());
  }
  
  // REMOVED - 69-feature logging block (not needed for 35-feature model)
  // The 35 features are already logged in the features_array block above
  
  // ============================================================================
  // CRITICAL: Print features BEFORE passing to Python model
  // This is AFTER all preprocessing (log10 transform, etc.) and BEFORE model input
  // Updated for 35-feature model
  // ============================================================================
  static const char* feature_names_for_print[35] = {
    "first_active_rank",                // 0
    "first_passive_rank",               // 1
    "active_rank_normalized",           // 2
    "passive_rank_normalized",          // 3
    "first_active_score",               // 4
    "first_passive_score",              // 5
    "level_avg_score",                  // 6
    "score_rank_interaction",           // 7
    "key_range_start",                  // 8
    "key_range_end",                    // 9
    "key_range_size",                   // 10
    "log10_key_range_size",             // 11
    "level0_cumulative_compaction_count",  // 12
    "level1_cumulative_compaction_count",  // 13
    "level2_cumulative_compaction_count",  // 14
    "level2_total_size",                // 15
    "level4_current_file_count",        // 16
    "level4_cumulative_compaction_count",  // 17
    "level5_current_file_count",        // 18
    "level5_cumulative_compaction_count",  // 19
    "level5_total_size",                // 20
    "overlap_with_lower",               // 21
    "overlap_with_upper",               // 22
    "overlap_count_with_lower",         // 23
    "overlap_count_with_upper",         // 24
    "overlap_ratio_with_lower",         // 25
    "overlap_ratio_with_upper",         // 26
    "left_neighbor_key_distance",       // 27
    "right_neighbor_key_distance",      // 28
    "min_neighbor_key_distance",        // 29
    "avg_neighbor_key_distance",        // 30
    "better_score_files_count",         // 31
    "worse_score_files_count",          // 32
    "competition_ratio",                // 33
    "urgency_score",                    // 34
  };
  
  // ALWAYS print features before model input using RocksDB LOG (writes to db1/LOG)
  if (info_log != nullptr) {
    double min_feature = features_array[0], max_feature = features_array[0];
    for (size_t i = 0; i < 35; i++) {
      if (!std::isnan(features_array[i]) && !std::isinf(features_array[i])) {
        if (features_array[i] < min_feature) min_feature = features_array[i];
        if (features_array[i] > max_feature) max_feature = features_array[i];
      }
    }
    
    // Log header
    ROCKS_LOG_INFO(info_log, "[ML FEATURES TO MODEL] ==========================================");
    ROCKS_LOG_INFO(info_log, "[ML FEATURES TO MODEL] level=%d, file_number=%" PRIu64, level, file_number);
    ROCKS_LOG_INFO(info_log, "[ML FEATURES TO MODEL] Features (35 features, BEFORE model input):");
    
    // Log each feature
    for (size_t i = 0; i < 35; i++) {
      if (std::isnan(features_array[i]) || std::isinf(features_array[i])) {
        ROCKS_LOG_ERROR(info_log, "[ML FEATURES TO MODEL] ERROR: Invalid feature[%zu]=%.6f (NaN/Inf) - %s", 
                        i, features_array[i], feature_names_for_print[i]);
      } else {
        ROCKS_LOG_INFO(info_log, "[ML FEATURES TO MODEL]   [%2zu] %-40s = %15.6f", 
                       i, feature_names_for_print[i], features_array[i]);
      }
    }
    
    // Log summary
    ROCKS_LOG_INFO(info_log, "[ML FEATURES TO MODEL] Feature range: min=%.6f, max=%.6f", min_feature, max_feature);
    ROCKS_LOG_INFO(info_log, "[ML FEATURES TO MODEL] ==========================================");
  }
  
  // Predict using level-specific Python model (69 features) 
  double predicted_lifetime =
      ROCKSDB_NAMESPACE::PredictFileLifetimePythonByLevel(features_array, 69,
                                                          level_for_predict);
  
  // Log prediction result to db1/LOG
  if (info_log) {
    ROCKS_LOG_INFO(info_log,
                   "[ML Prediction] file_number=%" PRIu64 " level=%d predicted_lifetime=%.6f",
                   file_number, level, predicted_lifetime);
  }
  
  return predicted_lifetime;
}
#endif  // ROCKSDB_ML_PREDICT_PYTHON

// Map predicted lifetime to WriteLifeTimeHint
// Mapping strategy:
// - Level 0: Always use WLTH_LEVEL0
// - Based on predicted lifetime:
//   - < 10s:   WLTH_LEVEL0
//   - 10-50s:  WLTH_LEVEL1
//   - 50-100s: WLTH_LEVEL2
//   - 100-150s: WLTH_LEVEL3
//   - 150-200s: WLTH_LEVEL4
//   - 200-400s: WLTH_LEVEL5
//   - > 400s:   WLTH_LEVEL6
// Convert MLFeatures to 35-element double array (matching training order)
// Reduced from 69 features based on feature importance analysis
// MLFeaturesToArray is declared in event_helpers_ml_features.h
// Converts MLFeatures to 69-element double array matching training data order
void MLFeaturesToArray(const MLFeatures& features, double* features_array, size_t array_size) {
  if (array_size < 69) {
    return;  // Array too small
  }
  
  // Rank/Score features (11 features, indices 0-10)
  features_array[0] = static_cast<double>(features.first_active_rank);
  features_array[1] = static_cast<double>(features.first_passive_rank);
  features_array[2] = features.active_rank_normalized;
  features_array[3] = features.passive_rank_normalized;
  features_array[4] = features.file_count_ratio;
  features_array[5] = features.first_active_score;
  features_array[6] = features.first_passive_score;
  features_array[7] = features.level_avg_score;
  features_array[8] = features.score_rank_interaction;
  features_array[9] = static_cast<double>(features.rank_difference);
  features_array[10] = features.score_difference;
  
  // Key Range features (6 features, indices 11-16)
  features_array[11] = static_cast<double>(features.key_range_start);
  features_array[12] = static_cast<double>(features.key_range_end);
  features_array[13] = static_cast<double>(features.key_range_size);
  features_array[14] = features.log10_key_range_size;
  features_array[15] = features.key_range_position_in_level;
  features_array[16] = features.key_range_percentile_in_level;
  
  // Level Stats features (33 features, indices 17-49)
  // Level 0
  features_array[17] = static_cast<double>(features.level0_current_file_count);
  features_array[18] = static_cast<double>(features.level0_cumulative_compaction_count);
  features_array[19] = static_cast<double>(features.level0_cumulative_file_count);
  // Level 1
  features_array[20] = static_cast<double>(features.level1_current_file_count);
  features_array[21] = static_cast<double>(features.level1_cumulative_compaction_count);
  features_array[22] = static_cast<double>(features.level1_cumulative_file_count);
  features_array[23] = static_cast<double>(features.level1_cumulative_trivial_move_count);
  features_array[24] = static_cast<double>(features.level1_total_size);
  // Level 2
  features_array[25] = static_cast<double>(features.level2_current_file_count);
  features_array[26] = static_cast<double>(features.level2_cumulative_compaction_count);
  features_array[27] = static_cast<double>(features.level2_cumulative_file_count);
  features_array[28] = static_cast<double>(features.level2_cumulative_trivial_move_count);
  features_array[29] = static_cast<double>(features.level2_total_size);
  // Level 3
  features_array[30] = static_cast<double>(features.level3_current_file_count);
  features_array[31] = static_cast<double>(features.level3_cumulative_compaction_count);
  features_array[32] = static_cast<double>(features.level3_cumulative_file_count);
  features_array[33] = static_cast<double>(features.level3_cumulative_trivial_move_count);
  features_array[34] = static_cast<double>(features.level3_total_size);
  // Level 4
  features_array[35] = static_cast<double>(features.level4_current_file_count);
  features_array[36] = static_cast<double>(features.level4_cumulative_compaction_count);
  features_array[37] = static_cast<double>(features.level4_cumulative_file_count);
  features_array[38] = static_cast<double>(features.level4_cumulative_trivial_move_count);
  features_array[39] = static_cast<double>(features.level4_total_size);
  // Level 5
  features_array[40] = static_cast<double>(features.level5_current_file_count);
  features_array[41] = static_cast<double>(features.level5_cumulative_compaction_count);
  features_array[42] = static_cast<double>(features.level5_cumulative_file_count);
  features_array[43] = static_cast<double>(features.level5_cumulative_trivial_move_count);
  features_array[44] = static_cast<double>(features.level5_total_size);
  // Level 6
  features_array[45] = static_cast<double>(features.level6_current_file_count);
  features_array[46] = static_cast<double>(features.level6_cumulative_compaction_count);
  features_array[47] = static_cast<double>(features.level6_cumulative_file_count);
  features_array[48] = static_cast<double>(features.level6_cumulative_trivial_move_count);
  features_array[49] = static_cast<double>(features.level6_total_size);
  
  // Overlap features (6 features, indices 50-55)
  features_array[50] = static_cast<double>(features.overlap_with_lower);
  features_array[51] = static_cast<double>(features.overlap_with_upper);
  features_array[52] = static_cast<double>(features.overlap_count_with_lower);
  features_array[53] = static_cast<double>(features.overlap_count_with_upper);
  features_array[54] = features.overlap_ratio_with_lower;
  features_array[55] = features.overlap_ratio_with_upper;
  
  // Neighbor features (4 features, indices 56-59)
  features_array[56] = features.left_neighbor_key_distance;
  features_array[57] = features.right_neighbor_key_distance;
  features_array[58] = features.min_neighbor_key_distance;
  features_array[59] = features.avg_neighbor_key_distance;
  
  // Competition features (4 features, indices 60-63)
  features_array[60] = static_cast<double>(features.better_score_files_count);
  features_array[61] = static_cast<double>(features.worse_score_files_count);
  features_array[62] = features.competition_ratio;
  features_array[63] = static_cast<double>(features.neighbor_files_count);
  
  // Other features (5 features, indices 64-68)
  features_array[64] = features.lower_level_capacity_ratio;
  features_array[65] = features.upper_level_capacity_ratio;
  features_array[66] = features.urgency_score;
  features_array[67] = features.health_score;
  features_array[68] = features.stability_score;
}

Env::WriteLifeTimeHint MapLifetimeToHint(double predicted_lifetime, int level) {
  // Level 0 always uses WLTH_LEVEL0.
  if (level == 0) {
    return Env::WLTH_LEVEL0;
  }

  // If prediction failed (lifetime <= 0) or invalid, fallback to level-based hint.
  if (std::isnan(predicted_lifetime) || std::isinf(predicted_lifetime) || predicted_lifetime <= 0.0) {
    if (level >= 0 && level <= 6) {
      return static_cast<Env::WriteLifeTimeHint>(
          static_cast<int>(Env::WLTH_LEVEL0) + level);
    }
    return Env::WLTH_LEVEL6;
  }

  // Map predicted lifetime to hint based on lifetime ranges.
  if (predicted_lifetime < 10.0) {
    return Env::WLTH_LEVEL0;
  } else if (predicted_lifetime < 50.0) {
    return Env::WLTH_LEVEL1;
  } else if (predicted_lifetime < 100.0) {
    return Env::WLTH_LEVEL2;
  } else if (predicted_lifetime < 150.0) {
    return Env::WLTH_LEVEL3;
  } else if (predicted_lifetime < 200.0) {
    return Env::WLTH_LEVEL4;
  } else if (predicted_lifetime < 400.0) {
    return Env::WLTH_LEVEL5;
  } else {
    return Env::WLTH_LEVEL6;
  }
}

// InitializeMLPredictorByLevel is declared in tools/ml_predict_python.h
bool InitializeMLPredictorByLevel() {
#ifdef ROCKSDB_ML_PREDICT_PYTHON
  return InitializePythonMLPredictor();
#else
  return false;
#endif
}

#ifdef ROCKSDB_ML_PREDICT_PYTHON
// PredictFileLifetimePythonByLevel is declared in tools/ml_predict_python.h
double PredictFileLifetimePythonByLevel(const double* features,
                                        size_t feature_count, int level) {
  bool init_ok = InitializePythonMLPredictor();
  if (!init_ok || !g_predict_func) {
    return 0.0;
  }

  int model_level = level;
  if (model_level < 1) {
    return 0.0;
  }
  if (model_level > 6) {
    model_level = 6;
  }

  // CRITICAL: Acquire GIL for thread-safe Python calls
  // RocksDB compaction runs in background threads, so we must acquire GIL
  PyGILState_STATE gstate = PyGILState_Ensure();
  
  PyObject* features_list = nullptr;
  PyObject* level_obj = nullptr;
  PyObject* args = nullptr;
  PyObject* result = nullptr;
  double predicted_lifetime = -1.0;

  // Build features list
  features_list = PyList_New(feature_count);
  if (!features_list) {
    fprintf(stderr, "[ERROR] PredictFileLifetimePythonByLevel: PyList_New failed\n");
    fflush(stderr);
    PyGILState_Release(gstate);
    return -1.0;
  }
  
  for (size_t i = 0; i < feature_count; ++i) {
    PyObject* item = PyFloat_FromDouble(features[i]);
    if (!item) {
      fprintf(stderr, "[ERROR] PredictFileLifetimePythonByLevel: PyFloat_FromDouble failed at index %zu\n", i);
      fflush(stderr);
      Py_DECREF(features_list);
      PyGILState_Release(gstate);
      return -1.0;
    }
    PyList_SetItem(features_list, i, item);
  }

  // Build arguments tuple
  level_obj = PyLong_FromLong(model_level);
  if (!level_obj) {
    fprintf(stderr, "[ERROR] PredictFileLifetimePythonByLevel: PyLong_FromLong failed\n");
    fflush(stderr);
    Py_DECREF(features_list);
    PyGILState_Release(gstate);
    return -1.0;
  }
  
  args = PyTuple_New(2);
  if (!args) {
    fprintf(stderr, "[ERROR] PredictFileLifetimePythonByLevel: PyTuple_New failed\n");
    fflush(stderr);
    Py_DECREF(features_list);
    Py_DECREF(level_obj);
    PyGILState_Release(gstate);
    return -1.0;
  }
  
  PyTuple_SetItem(args, 0, features_list);  // features first (matching Python function signature)
  PyTuple_SetItem(args, 1, level_obj);  // level second

  result = PyObject_CallObject(g_predict_func, args);
  
  Py_DECREF(args);

  if (!result) {
    fprintf(stderr, "[ERROR] PredictFileLifetimePythonByLevel: Python调用失败！\n");
    fflush(stderr);
    if (PyErr_Occurred()) {
      PyErr_Print();
      PyErr_Clear();
    }
    PyGILState_Release(gstate);
    return -1.0;
  }

  predicted_lifetime = PyFloat_AsDouble(result);
  
  if (PyErr_Occurred()) {
    fprintf(stderr, "[ERROR] PredictFileLifetimePythonByLevel: 返回值转换时发生Python错误！\n");
    fflush(stderr);
    PyErr_Print();
    PyErr_Clear();
    Py_DECREF(result);
    PyGILState_Release(gstate);
    return -1.0;
  }
  
  Py_DECREF(result);
  PyGILState_Release(gstate);
  
  return predicted_lifetime;
}
#endif  // ROCKSDB_ML_PREDICT_PYTHON

}  // namespace ROCKSDB_NAMESPACE

