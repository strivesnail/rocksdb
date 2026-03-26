// ONNX Runtime based predictor implementation

#include "rocksdb/rocksdb_namespace.h"
#include "db/onnx_predictor.h"
#include "db/scaler_params.h"

#include <cstring>
#include <iostream>
#include <fstream>

// Disable shadow warnings for ONNX Runtime headers (they have internal shadowing)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wshadow"

// ONNX Runtime headers
#include <onnxruntime_cxx_api.h>

#pragma GCC diagnostic pop

namespace ROCKSDB_NAMESPACE {

namespace onnx_internal {

// Internal structure to hold ONNX Runtime session
struct OnnxSession {
  Ort::Session session;
  Ort::AllocatorWithDefaultOptions allocator;
  std::string input_name;
  std::string output_name;
  
  OnnxSession(Ort::Env& env, const char* model_path, Ort::SessionOptions& options)
      : session(env, model_path, options) {
    // Get input/output names
    input_name = session.GetInputNameAllocated(0, allocator).get();
    output_name = session.GetOutputNameAllocated(0, allocator).get();
  }
};

}  // namespace onnx_internal

// Global ONNX Runtime environment (created once)
static std::unique_ptr<Ort::Env> g_ort_env;
static std::mutex g_env_mutex;

static Ort::Env& GetOrtEnv() {
  std::lock_guard<std::mutex> lock(g_env_mutex);
  if (!g_ort_env) {
    g_ort_env = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "RocksDB_ONNX");
  }
  return *g_ort_env;
}

OnnxPredictor::OnnxPredictor() : initialized_(false) {}

OnnxPredictor::~OnnxPredictor() {
  // Do NOT call Shutdown() here automatically.
  // The static singleton may be destroyed after DBImpl destructor starts
  // but before all background threads finish. Let TwoPhaseWriteManager
  // explicitly call ShutdownOnnxPredictor() when it's safe.
  // Shutdown();  // Disabled - causes crash during program exit
}

OnnxPredictor& OnnxPredictor::Instance() {
  static OnnxPredictor instance;
  return instance;
}

bool OnnxPredictor::Initialize(const std::string& models_dir) {
  std::lock_guard<std::mutex> lock(init_mutex_);
  
  if (initialized_) {
    return true;
  }
  
  models_dir_ = models_dir;
  
  try {
    Ort::Env& env = GetOrtEnv();
    
    // Session options
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);  // Single thread per inference
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    
    // Load models for Level 1-5
    int loaded_count = 0;
    for (int level = 1; level <= 5; ++level) {
      std::string model_path = models_dir_ + "/model_level_" + std::to_string(level) + ".onnx";
      
      // Check if file exists
      std::ifstream f(model_path);
      if (!f.good()) {
        std::cerr << "[ONNX] Model file not found: " << model_path << std::endl;
        continue;
      }
      f.close();
      
      try {
        sessions_[level] = std::make_unique<onnx_internal::OnnxSession>(
            env, model_path.c_str(), session_options);
        loaded_count++;
        std::cerr << "[ONNX] Loaded model for Level " << level 
                  << " (input: " << sessions_[level]->input_name 
                  << ", output: " << sessions_[level]->output_name << ")" << std::endl;
      } catch (const Ort::Exception& e) {
        std::cerr << "[ONNX] Failed to load model for Level " << level 
                  << ": " << e.what() << std::endl;
      }
    }
    
    if (loaded_count == 0) {
      std::cerr << "[ONNX] No models loaded!" << std::endl;
      return false;
    }
    
    initialized_ = true;
    std::cerr << "[ONNX] Predictor initialized with " << loaded_count << " models" << std::endl;
    return true;
    
  } catch (const std::exception& e) {
    std::cerr << "[ONNX] Initialization failed: " << e.what() << std::endl;
    return false;
  }
}

void OnnxPredictor::Shutdown() {
  std::lock_guard<std::mutex> lock(init_mutex_);
  
  for (int i = 0; i < 6; ++i) {
    sessions_[i].reset();
  }
  
  initialized_ = false;
}

void OnnxPredictor::ApplyScaler(const double* input, float* output, int level) {
  const double* mean = GetScalerMean(level);
  const double* scale = GetScalerScale(level);
  
  if (!mean || !scale) {
    // No scaler for this level, just convert to float
    for (size_t i = 0; i < kNumFeatures; ++i) {
      output[i] = static_cast<float>(input[i]);
    }
    return;
  }
  
  // StandardScaler transform: (x - mean) / scale
  for (size_t i = 0; i < kNumFeatures; ++i) {
    output[i] = static_cast<float>((input[i] - mean[i]) / scale[i]);
  }
}

double OnnxPredictor::Predict(const double* features, size_t feature_count, int level) {
  if (!initialized_) {
    std::cerr << "[ONNX] Predictor not initialized" << std::endl;
    return -1.0;
  }
  
  // Validate level
  if (level < 1 || level > 5) {
    std::cerr << "[ONNX] Invalid level: " << level << " (must be 1-5)" << std::endl;
    return -1.0;
  }
  
  // Check if model is loaded for this level
  if (!sessions_[level]) {
    std::cerr << "[ONNX] No model loaded for Level " << level << std::endl;
    return -1.0;
  }
  
  // Validate feature count
  if (feature_count != kNumFeatures) {
    std::cerr << "[ONNX] Invalid feature count: " << feature_count 
              << " (expected " << kNumFeatures << ")" << std::endl;
    return -1.0;
  }
  
  // Lock for thread-safe inference on this level
  std::lock_guard<std::mutex> lock(predict_mutex_[level]);
  
  try {
    // Apply scaler transform
    std::vector<float> scaled_features(kNumFeatures);
    ApplyScaler(features, scaled_features.data(), level);
    
    // Create input tensor
    std::vector<int64_t> input_shape = {1, static_cast<int64_t>(kNumFeatures)};
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(
        OrtArenaAllocator, OrtMemTypeDefault);
    
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info, scaled_features.data(), kNumFeatures,
        input_shape.data(), input_shape.size());
    
    // Run inference
    const char* input_names[] = {sessions_[level]->input_name.c_str()};
    const char* output_names[] = {sessions_[level]->output_name.c_str()};
    
    auto output_tensors = sessions_[level]->session.Run(
        Ort::RunOptions{nullptr},
        input_names, &input_tensor, 1,
        output_names, 1);
    
    // Get output
    float* output_data = output_tensors[0].GetTensorMutableData<float>();
    double prediction = static_cast<double>(output_data[0]);
    
    // Ensure non-negative
    return std::max(0.0, prediction);
    
  } catch (const Ort::Exception& e) {
    std::cerr << "[ONNX] Prediction failed for Level " << level 
              << ": " << e.what() << std::endl;
    return -1.0;
  } catch (const std::exception& e) {
    std::cerr << "[ONNX] Prediction error: " << e.what() << std::endl;
    return -1.0;
  }
}

// C-style API implementation
double PredictFileLifetimeONNX(const double* features, size_t feature_count, int level) {
  return OnnxPredictor::Instance().Predict(features, feature_count, level);
}

bool InitializeOnnxPredictor(const char* models_dir) {
  return OnnxPredictor::Instance().Initialize(models_dir ? models_dir : "");
}

void ShutdownOnnxPredictor() {
  OnnxPredictor::Instance().Shutdown();
}

}  // namespace ROCKSDB_NAMESPACE
