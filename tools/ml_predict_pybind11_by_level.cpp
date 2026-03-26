// pybind11 wrapper for ML prediction by level
// Compile with: g++ -O3 -shared -std=c++14 -fPIC `python3 -m pybind11 --includes` ml_predict_pybind11_by_level.cpp -o ml_predict_by_level.so `python3-config --ldflags`

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <vector>
#include <string>

namespace py = pybind11;

// Python module for ML prediction by level
class MLPredictorByLevel {
public:
    MLPredictorByLevel() {
        // Import Python module
        py::module_ ml_module = py::module_::import("ml_predict_lifetime_by_level");
        
        // Get the predict function
        predict_func_ = ml_module.attr("predict_file_lifetime_by_level");
    }
    
    double predict(const std::vector<double>& features, int level) {
        // Convert features to numpy array
        py::array_t<double> features_array(features.size(), features.data());
        
        // Call Python predict function
        py::object result = predict_func_(features_array, level);
        
        // Convert result to double
        return result.cast<double>();
    }
    
private:
    py::object predict_func_;
};

// Global predictor instance (loaded once)
static std::unique_ptr<MLPredictorByLevel> g_predictor_by_level;

// Initialize predictor (call once at startup)
extern "C" bool InitializeMLPredictorByLevel() {
    try {
        if (!g_predictor_by_level) {
            g_predictor_by_level = std::make_unique<MLPredictorByLevel>();
        }
        return true;
    } catch (const std::exception& e) {
        return false;
    }
}

// Predict function (called from C++)
extern "C" double PredictFileLifetimePythonByLevel(const double* features,
                                                    size_t feature_count, 
                                                    int level) {
    if (!g_predictor_by_level) {
        if (!InitializeMLPredictorByLevel()) {
            return 0.0;  // Predictor initialization failed
        }
    }
    
    try {
        std::vector<double> features_vec(features, features + feature_count);
        return g_predictor_by_level->predict(features_vec, level);
    } catch (const std::exception& e) {
        return 0.0;  // Prediction failed
    }
}

// Cleanup (call at shutdown)
extern "C" void CleanupMLPredictorByLevel() {
    g_predictor_by_level.reset();
}



















