//  Copyright (c) 2011-present, Facebook, Inc.  All rights reserved.
//  This source code is licensed under both the GPLv2 (found in the
//  COPYING file in the root directory) and Apache 2.0 License
//  (found in the LICENSE.Apache file in the root directory).

#include "db/ml_prediction_process_pool.h"

#include <unistd.h>
#include <sys/wait.h>
#include <sys/types.h>
#include <cstring>
#include <cstdio>
#include <cerrno>

#ifdef ROCKSDB_ML_PREDICT_PYTHON
#include <Python.h>
#endif

#include "logging/logging.h"
#include "rocksdb/env.h"

namespace ROCKSDB_NAMESPACE {


std::unique_ptr<MLPredictionProcessPool> g_ml_prediction_pool;

MLPredictionProcessPool::MLPredictionProcessPool()
    : initialized_(false), num_workers_(0), next_request_id_(1), current_worker_(0) {
}

MLPredictionProcessPool::~MLPredictionProcessPool() {
  Shutdown();
}

Status MLPredictionProcessPool::Initialize(int num_workers) {
  if (initialized_.load()) {
    return Status::OK();
  }

  if (num_workers <= 0) {
    num_workers = 2;
  }
  if (num_workers > 8) {
    num_workers = 8;
  }

  num_workers_ = num_workers;
  workers_.resize(num_workers_);


  for (int i = 0; i < num_workers_; ++i) {
    Status s = CreateWorkerProcess(i, &workers_[i]);
    if (!s.ok()) {
      Shutdown();
      return s;
    }
  }

  initialized_.store(true);
  return Status::OK();
}

void MLPredictionProcessPool::Shutdown() {
  if (!initialized_.load()) {
    return;
  }

  initialized_.store(false);


  for (auto& worker : workers_) {
    if (worker.active && worker.pid > 0) {

      if (worker.request_pipe[0] >= 0) close(worker.request_pipe[0]);
      if (worker.request_pipe[1] >= 0) close(worker.request_pipe[1]);
      if (worker.response_pipe[0] >= 0) close(worker.response_pipe[0]);
      if (worker.response_pipe[1] >= 0) close(worker.response_pipe[1]);


      if (kill(worker.pid, SIGTERM) == 0) {
        int status;
        waitpid(worker.pid, &status, 0);
      }
      worker.active = false;
      worker.pid = -1;
    }
  }

  workers_.clear();
  num_workers_ = 0;
}

Status MLPredictionProcessPool::CreateWorkerProcess(int worker_id, WorkerProcess* worker) {

  if (pipe(worker->request_pipe) != 0) {
    return Status::IOError("Failed to create request pipe", strerror(errno));
  }


  if (pipe(worker->response_pipe) != 0) {
    close(worker->request_pipe[0]);
    close(worker->request_pipe[1]);
    return Status::IOError("Failed to create response pipe", strerror(errno));
  }


  pid_t pid = fork();
  if (pid < 0) {
    close(worker->request_pipe[0]);
    close(worker->request_pipe[1]);
    close(worker->response_pipe[0]);
    close(worker->response_pipe[1]);
    return Status::IOError("Failed to fork worker process", strerror(errno));
  }

  if (pid == 0) {

    close(worker->request_pipe[1]);
    close(worker->response_pipe[0]);


    WorkerMain(worker_id, worker->request_pipe[0], worker->response_pipe[1]);


    _exit(1);
  } else {

    close(worker->request_pipe[0]);
    close(worker->response_pipe[1]);

    worker->pid = pid;
    worker->active = true;
    return Status::OK();
  }
}

void MLPredictionProcessPool::WorkerMain(int /*worker_id*/, int request_fd, int response_fd) {
#ifdef ROCKSDB_ML_PREDICT_PYTHON

  signal(SIGPIPE, SIG_IGN);


  if (!Py_IsInitialized()) {
    Py_Initialize();
    if (!Py_IsInitialized()) {
      return;
    }
  }


  const char* models_path = std::getenv("ROCKSDB_ML_MODELS_PATH");
  if (models_path) {
    PyObject* env_dict = PySys_GetObject("environ");
    if (env_dict) {
      PyObject* key = PyUnicode_FromString("ROCKSDB_ML_MODELS_PATH");
      PyObject* value = PyUnicode_FromString(models_path);
      if (key && value) {
        PyDict_SetItem(env_dict, key, value);
      }
      Py_XDECREF(key);
      Py_XDECREF(value);
    }
  }


  const char* tools_path = std::getenv("ROCKSDB_TOOLS_PATH");
  if (tools_path) {
    PyObject* sys_path = PySys_GetObject("path");
    if (sys_path) {
      PyObject* path_obj = PyUnicode_FromString(tools_path);
      if (path_obj) {
        PyList_Append(sys_path, path_obj);
        Py_DECREF(path_obj);
      }
    }
  }


  PyObject* module = PyImport_ImportModule("ml_predict_lifetime_by_level");
  if (!module) {
    PyErr_Print();
    return;
  }


  PyObject* predict_func = PyObject_GetAttrString(module, "predict_file_lifetime_by_level");
  if (!predict_func || !PyCallable_Check(predict_func)) {
    Py_XDECREF(predict_func);
    Py_DECREF(module);
    return;
  }


  PyObject* init_func = PyObject_GetAttrString(module, "initialize");
  if (init_func && PyCallable_Check(init_func)) {
    PyObject* result = PyObject_CallObject(init_func, nullptr);
    Py_XDECREF(result);
  }
  Py_XDECREF(init_func);


  while (true) {

    PredictionRequest request;
    ssize_t n = read(request_fd, &request.request_id, sizeof(request.request_id));
    if (n <= 0) {
      break;
    }
    if (n != sizeof(request.request_id)) {
      break;
    }

    n = read(request_fd, &request.level, sizeof(request.level));
    if (n <= 0 || n != sizeof(request.level)) break;

    size_t feature_count = 0;
    n = read(request_fd, &feature_count, sizeof(feature_count));
    if (n <= 0 || n != sizeof(feature_count)) break;

    if (feature_count == 0 || feature_count > 1000) {

      PredictionResponse error_response(request.request_id, 0.0, false);
      ssize_t written = write(response_fd, &error_response.request_id, sizeof(error_response.request_id));
      if (written != sizeof(error_response.request_id)) break;
      written = write(response_fd, &error_response.predicted_lifetime, sizeof(error_response.predicted_lifetime));
      if (written != sizeof(error_response.predicted_lifetime)) break;
      written = write(response_fd, &error_response.success, sizeof(error_response.success));
      if (written != sizeof(error_response.success)) break;
      continue;
    }

    request.features.resize(feature_count);
    n = read(request_fd, request.features.data(), feature_count * sizeof(double));
    if (n <= 0 || n != static_cast<ssize_t>(feature_count * sizeof(double))) break;


    double predicted_lifetime = 0.0;
    bool success = false;

    if (feature_count == 69) {

      PyObject* features_list = PyList_New(feature_count);
      if (features_list) {
        for (size_t i = 0; i < feature_count; ++i) {
          PyObject* item = PyFloat_FromDouble(request.features[i]);
          if (item) {
            PyList_SetItem(features_list, i, item);
          }
        }

        PyObject* level_obj = PyLong_FromLong(request.level);
        PyObject* args = PyTuple_New(2);
        PyTuple_SetItem(args, 0, features_list);
        PyTuple_SetItem(args, 1, level_obj);

        PyObject* result = PyObject_CallObject(predict_func, args);
        Py_DECREF(args);

        if (result) {
          predicted_lifetime = PyFloat_AsDouble(result);
          if (!PyErr_Occurred()) {
            success = true;
          } else {
            PyErr_Clear();
          }
          Py_DECREF(result);
        } else {
          PyErr_Clear();
        }
      }
    }


    PredictionResponse response(request.request_id, predicted_lifetime, success);
    ssize_t written = write(response_fd, &response.request_id, sizeof(response.request_id));
    if (written != sizeof(response.request_id)) break;
    written = write(response_fd, &response.predicted_lifetime, sizeof(response.predicted_lifetime));
    if (written != sizeof(response.predicted_lifetime)) break;
    written = write(response_fd, &response.success, sizeof(response.success));
    if (written != sizeof(response.success)) break;
  }


  Py_DECREF(predict_func);
  Py_DECREF(module);
  Py_Finalize();
#else

  (void)request_fd;
  (void)response_fd;
#endif
}

Status MLPredictionProcessPool::SendRequest(int worker_id, const PredictionRequest& request) {
  if (worker_id < 0 || worker_id >= num_workers_ || !workers_[worker_id].active) {
    return Status::InvalidArgument("Invalid worker ID");
  }

  const WorkerProcess& worker = workers_[worker_id];


  ssize_t n = write(worker.request_pipe[1], &request.request_id, sizeof(request.request_id));
  if (n != sizeof(request.request_id)) {
    return Status::IOError("Failed to write request_id");
  }

  n = write(worker.request_pipe[1], &request.level, sizeof(request.level));
  if (n != sizeof(request.level)) {
    return Status::IOError("Failed to write level");
  }

  size_t feature_count = request.features.size();
  n = write(worker.request_pipe[1], &feature_count, sizeof(feature_count));
  if (n != sizeof(feature_count)) {
    return Status::IOError("Failed to write feature_count");
  }

  n = write(worker.request_pipe[1], request.features.data(), feature_count * sizeof(double));
  if (n != static_cast<ssize_t>(feature_count * sizeof(double))) {
    return Status::IOError("Failed to write features");
  }

  return Status::OK();
}

Status MLPredictionProcessPool::ReceiveResponse(int worker_id, PredictionResponse* response) {
  if (worker_id < 0 || worker_id >= num_workers_ || !workers_[worker_id].active) {
    return Status::InvalidArgument("Invalid worker ID");
  }

  const WorkerProcess& worker = workers_[worker_id];


  ssize_t n = read(worker.response_pipe[0], &response->request_id, sizeof(response->request_id));
  if (n != sizeof(response->request_id)) {
    return Status::IOError("Failed to read request_id");
  }

  n = read(worker.response_pipe[0], &response->predicted_lifetime, sizeof(response->predicted_lifetime));
  if (n != sizeof(response->predicted_lifetime)) {
    return Status::IOError("Failed to read predicted_lifetime");
  }

  n = read(worker.response_pipe[0], &response->success, sizeof(response->success));
  if (n != sizeof(response->success)) {
    return Status::IOError("Failed to read success");
  }

  return Status::OK();
}

int MLPredictionProcessPool::SelectWorker() {

  int worker = current_worker_.fetch_add(1) % num_workers_;
  return worker;
}

double MLPredictionProcessPool::Predict(int level, const double* features, size_t feature_count) {
  if (!initialized_.load() || num_workers_ == 0) {
    return 0.0;
  }


  int worker_id = SelectWorker();


  uint64_t request_id = next_request_id_.fetch_add(1);
  PredictionRequest request(request_id, level, std::vector<double>(features, features + feature_count));


  Status s = SendRequest(worker_id, request);
  if (!s.ok()) {
    return 0.0;
  }


  PredictionResponse response;
  s = ReceiveResponse(worker_id, &response);
  if (!s.ok() || !response.success || response.request_id != request_id) {
    return 0.0;
  }

  return response.predicted_lifetime;
}

}  // namespace ROCKSDB_NAMESPACE

