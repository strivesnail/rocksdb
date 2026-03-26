//  Copyright (c) 2011-present, Facebook, Inc.  All rights reserved.
//  This source code is licensed under both the GPLv2 (found in the
//  COPYING file in the root directory) and Apache 2.0 License
//  (found in the LICENSE.Apache file in the root directory).

#pragma once

#include "rocksdb/env.h"
#include "rocksdb/status.h"
#include "rocksdb/file_system.h"
#include "rocksdb/io_status.h"
#include <string>
#include <vector>

namespace ROCKSDB_NAMESPACE {

// 内存中的 FSWritableFile 实现
// 用于两阶段写入：第一次写入内存，然后根据预测结果写入最终位置
class InMemoryWritableFile : public FSWritableFile {
 public:
  InMemoryWritableFile() : FSWritableFile(), buffer_(), closed_(false) {}
  
  ~InMemoryWritableFile() override = default;
  
  // Append data to the end of the file
  IOStatus Append(const Slice& data, const IOOptions& /*options*/,
                  IODebugContext* /*dbg*/) override {
    if (closed_) {
      return IOStatus::IOError("File is closed");
    }
    buffer_.append(data.data(), data.size());
    return IOStatus::OK();
  }
  
  IOStatus Append(const Slice& data, const IOOptions& options,
                  const DataVerificationInfo& /* verification_info */,
                  IODebugContext* dbg) override {
    return Append(data, options, dbg);
  }
  
  // PositionedAppend (not used in our case)
  IOStatus PositionedAppend(const Slice& /* data */, uint64_t /* offset */,
                            const IOOptions& /*options*/,
                            IODebugContext* /*dbg*/) override {
    return IOStatus::NotSupported("PositionedAppend not supported");
  }
  
  IOStatus PositionedAppend(const Slice& /* data */, uint64_t /* offset */,
                                const IOOptions& /*options*/,
                                const DataVerificationInfo& /* verification_info */,
                                IODebugContext* /*dbg*/) override {
    return IOStatus::NotSupported("PositionedAppend not supported");
  }
  
  // Truncate (not used in our case)
  IOStatus Truncate(uint64_t /* size */, const IOOptions& /*options*/,
                    IODebugContext* /*dbg*/) override {
    return IOStatus::OK();
  }
  
  // Close the file
  IOStatus Close(const IOOptions& /*options*/, IODebugContext* /*dbg*/) override {
    closed_ = true;
    return IOStatus::OK();
  }
  
  // Flush (no-op for in-memory file)
  IOStatus Flush(const IOOptions& /*options*/, IODebugContext* /*dbg*/) override {
    return IOStatus::OK();
  }
  
  // Sync (no-op for in-memory file)
  IOStatus Sync(const IOOptions& /*options*/, IODebugContext* /*dbg*/) override {
    return IOStatus::OK();
  }
  
  // Fsync (no-op for in-memory file)
  IOStatus Fsync(const IOOptions& options, IODebugContext* dbg) override {
    return Sync(options, dbg);
  }
  
  // Get file size
  uint64_t GetFileSize(const IOOptions& /*options*/, IODebugContext* /*dbg*/) override {
    return buffer_.size();
  }
  
  // Get the buffer data
  const std::string& GetBuffer() const {
    return buffer_;
  }
  
  // Clear the buffer
  void Clear() {
    buffer_.clear();
    closed_ = false;
  }
  
  // Check if file is closed
  bool IsClosed() const {
    return closed_;
  }
  
 private:
  std::string buffer_;  // 内存 buffer
  bool closed_;         // 是否已关闭
};

}  // namespace ROCKSDB_NAMESPACE

