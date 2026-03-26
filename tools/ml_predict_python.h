//  Copyright (c) 2011-present, Facebook, Inc.  All rights reserved.
//  This source code is licensed under both the GPLv2 (found in the
//  COPYING file in the root directory) and Apache 2.0 License
//  (found in the LICENSE.Apache file in the root directory).
//
// Copyright (c) 2011 The LevelDB Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file. See the AUTHORS file for names of contributors.

#pragma once

#include <cstddef>

namespace ROCKSDB_NAMESPACE {

// Initialize ML predictor (Python-based)
// Returns true if initialization succeeded, false otherwise
bool InitializeMLPredictorByLevel();

// Predict file lifetime using Python ML model
// features: array of feature values (35 features, reduced from 69)
// feature_count: number of features (should be 35)
// level: file level (1-6)
// Returns predicted lifetime in seconds, or 0.0 on error
double PredictFileLifetimePythonByLevel(const double* features,
                                        size_t feature_count, int level);

}  // namespace ROCKSDB_NAMESPACE

