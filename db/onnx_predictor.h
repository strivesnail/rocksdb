// ONNX Runtime based predictor for file lifetime prediction
// Replaces Python-based prediction to eliminate GIL bottleneck

#pragma once

#include "rocksdb/rocksdb_namespace.h"

#include <string>
#include <vector>
#include <memory>
#include <mutex>

namespace ROCKSDB_NAMESPACE {

// Forward declaration for ONNX Runtime types
namespace onnx_internal {
struct OnnxSession;
}

class OnnxPredictor {
 public:
  // Get singleton instance
  static OnnxPredictor& Instance();

  // Initialize the predictor with model directory
  // Returns true if initialization successful
  bool Initialize(const std::string& models_dir);

  // Check if predictor is initialized
  bool IsInitialized() const { return initialized_; }

  // Predict file lifetime for a given level
  // features: array of 69 features (raw, unscaled)
  // feature_count: number of features (should be 69)
  // level: target level (1-5)
  // Returns: predicted lifetime in seconds, or -1.0 on error
  double Predict(const double* features, size_t feature_count, int level);

  // Shutdown and release resources
  void Shutdown();

 private:
  OnnxPredictor();
  ~OnnxPredictor();

  // Disable copy
  OnnxPredictor(const OnnxPredictor&) = delete;
  OnnxPredictor& operator=(const OnnxPredictor&) = delete;

  // Apply StandardScaler transform: (x - mean) / scale
  void ApplyScaler(const double* input, float* output, int level);

  bool initialized_;
  std::string models_dir_;
  
  // ONNX Runtime sessions for each level (1-5)
  // Index 0 is unused, index 1-5 correspond to levels 1-5
  std::unique_ptr<onnx_internal::OnnxSession> sessions_[6];
  
  // Mutex for thread-safe initialization
  std::mutex init_mutex_;
  
  // Mutex for thread-safe inference (one per level to allow parallel inference on different levels)
  mutable std::mutex predict_mutex_[6];
};

// C-style function for easy integration (matches existing Python API)
// Returns predicted lifetime in seconds, or -1.0 on error
double PredictFileLifetimeONNX(const double* features, size_t feature_count, int level);

// Initialize ONNX predictor (call once at startup)
bool InitializeOnnxPredictor(const char* models_dir);

// Shutdown ONNX predictor (call at shutdown)
void ShutdownOnnxPredictor();

}  // namespace ROCKSDB_NAMESPACE
