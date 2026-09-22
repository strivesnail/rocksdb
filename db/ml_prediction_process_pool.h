//  Copyright (c) 2011-present, Facebook, Inc.  All rights reserved.
//  This source code is licensed under both the GPLv2 (found in the
//  COPYING file in the root directory) and Apache 2.0 License
//  (found in the LICENSE.Apache file in the root directory).

#pragma once

#include <atomic>
#include <memory>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>
#include <condition_variable>

#include "rocksdb/status.h"

namespace ROCKSDB_NAMESPACE {


struct PredictionRequest {
  uint64_t request_id;
  int level;
  std::vector<double> features;

  PredictionRequest() : request_id(0), level(0) {}
  PredictionRequest(uint64_t id, int l, const std::vector<double>& f)
      : request_id(id), level(l), features(f) {}
};


struct PredictionResponse {
  uint64_t request_id;
  double predicted_lifetime;
  bool success;

  PredictionResponse() : request_id(0), predicted_lifetime(0.0), success(false) {}
  PredictionResponse(uint64_t id, double lifetime, bool s)
      : request_id(id), predicted_lifetime(lifetime), success(s) {}
};



class MLPredictionProcessPool {
 public:
  MLPredictionProcessPool();
  ~MLPredictionProcessPool();


  Status Initialize(int num_workers = 2);


  void Shutdown();


  double Predict(int level, const double* features, size_t feature_count);


  bool IsInitialized() const { return initialized_.load(); }


  int GetNumWorkers() const { return num_workers_; }

 private:

  struct WorkerProcess {
    pid_t pid;
    int request_pipe[2];
    int response_pipe[2];
    bool active;

    WorkerProcess() : pid(-1), active(false) {
      request_pipe[0] = request_pipe[1] = -1;
      response_pipe[0] = response_pipe[1] = -1;
    }
  };


  Status CreateWorkerProcess(int worker_id, WorkerProcess* worker);


  static void WorkerMain(int worker_id, int request_fd, int response_fd);


  Status SendRequest(int worker_id, const PredictionRequest& request);


  Status ReceiveResponse(int worker_id, PredictionResponse* response);


  int SelectWorker();

  std::atomic<bool> initialized_;
  int num_workers_;
  std::vector<WorkerProcess> workers_;
  std::atomic<uint64_t> next_request_id_;
  std::atomic<int> current_worker_;

  mutable std::mutex mutex_;
};


extern std::unique_ptr<MLPredictionProcessPool> g_ml_prediction_pool;

}  // namespace ROCKSDB_NAMESPACE








