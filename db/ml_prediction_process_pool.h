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

// 预测请求结构
struct PredictionRequest {
  uint64_t request_id;
  int level;
  std::vector<double> features;
  
  PredictionRequest() : request_id(0), level(0) {}
  PredictionRequest(uint64_t id, int l, const std::vector<double>& f)
      : request_id(id), level(l), features(f) {}
};

// 预测响应结构
struct PredictionResponse {
  uint64_t request_id;
  double predicted_lifetime;
  bool success;
  
  PredictionResponse() : request_id(0), predicted_lifetime(0.0), success(false) {}
  PredictionResponse(uint64_t id, double lifetime, bool s)
      : request_id(id), predicted_lifetime(lifetime), success(s) {}
};

// ML 预测进程池管理器
// 使用多个工作进程来并行执行 Python ML 预测，避免 GIL 限制
class MLPredictionProcessPool {
 public:
  MLPredictionProcessPool();
  ~MLPredictionProcessPool();

  // 初始化进程池（创建工作进程）
  Status Initialize(int num_workers = 2);

  // 关闭进程池（清理所有工作进程）
  void Shutdown();

  // 执行预测（同步调用，等待结果）
  double Predict(int level, const double* features, size_t feature_count);

  // 检查是否已初始化
  bool IsInitialized() const { return initialized_.load(); }

  // 获取工作进程数量
  int GetNumWorkers() const { return num_workers_; }

 private:
  // 工作进程信息
  struct WorkerProcess {
    pid_t pid;
    int request_pipe[2];   // 父进程写，子进程读
    int response_pipe[2];  // 子进程写，父进程读
    bool active;
    
    WorkerProcess() : pid(-1), active(false) {
      request_pipe[0] = request_pipe[1] = -1;
      response_pipe[0] = response_pipe[1] = -1;
    }
  };

  // 创建工作进程
  Status CreateWorkerProcess(int worker_id, WorkerProcess* worker);

  // 工作进程主函数（在子进程中运行）
  static void WorkerMain(int worker_id, int request_fd, int response_fd);

  // 发送预测请求到工作进程
  Status SendRequest(int worker_id, const PredictionRequest& request);

  // 从工作进程接收预测响应
  Status ReceiveResponse(int worker_id, PredictionResponse* response);

  // 选择可用的工作进程（轮询）
  int SelectWorker();

  std::atomic<bool> initialized_;
  int num_workers_;
  std::vector<WorkerProcess> workers_;
  std::atomic<uint64_t> next_request_id_;
  std::atomic<int> current_worker_;  // 用于轮询选择工作进程
  
  mutable std::mutex mutex_;  // 保护共享状态
};

// 全局进程池实例（单例）
extern std::unique_ptr<MLPredictionProcessPool> g_ml_prediction_pool;

}  // namespace ROCKSDB_NAMESPACE








