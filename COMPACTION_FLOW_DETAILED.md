# RocksDB Compaction 完整流程详解

## 目录
1. [概述](#概述)
2. [核心数据结构](#核心数据结构)
3. [Compaction触发机制](#compaction触发机制)
4. [Compaction选择流程（PickCompaction）](#compaction选择流程pickcompaction)
5. [Compaction执行流程（CompactionJob）](#compaction执行流程compactionjob)
6. [状态变化与生命周期](#状态变化与生命周期)
7. [关键代码路径](#关键代码路径)

---

## 概述

RocksDB的Compaction是LSM-Tree的核心机制，负责：
- **合并多个SST文件**：将多个较小的文件合并成更大的文件
- **删除过期数据**：清理已删除的key和过期的数据
- **优化读取性能**：减少需要读取的文件数量
- **控制写入放大**：平衡写入性能和存储效率
- **维护LSM-Tree结构**：确保数据从上层level逐步下沉到底层level

---

## 核心数据结构

### 1. FileMetaData（文件元数据）

**位置**：`db/version_edit.h:195`

```cpp
struct FileMetaData {
  FileDescriptor fd;              // 文件描述符（文件号、路径ID、大小等）
  InternalKey smallest;            // 文件中最小的key
  InternalKey largest;             // 文件中最大的key
  
  // 文件统计信息
  uint64_t compensated_file_size; // 补偿后的文件大小（考虑删除条目）
  uint64_t num_entries;           // 条目总数（包括删除和范围删除）
  uint64_t num_deletions;        // 删除条目数
  uint64_t raw_key_size;          // 未压缩的key总大小
  uint64_t raw_value_size;        // 未压缩的value总大小
  
  // 状态标志
  int refs;                        // 引用计数
  bool being_compacted;            // 是否正在被compaction
  bool marked_for_compaction;      // 是否被标记为需要compaction
  Temperature temperature;         // 文件温度（热/温/冷）
  
  // 时间信息
  uint64_t file_creation_time;    // 文件创建时间
  uint64_t oldest_ancester_time;  // 最老祖先时间
  uint64_t epoch_number;           // 文件epoch编号
};
```

**关键字段说明**：
- `being_compacted`：防止同一文件被多个compaction同时处理
- `compensated_file_size`：用于计算compaction优先级，考虑删除条目后的实际大小
- `smallest/largest`：用于判断文件之间的key范围重叠

### 2. VersionStorageInfo（版本存储信息）

**位置**：`db/version_set.h:130`

```cpp
class VersionStorageInfo {
  // 每个level的文件列表
  std::vector<std::vector<FileMetaData*>> files_;
  
  // Compaction优先级排序的文件索引
  std::vector<std::vector<int>> files_by_compaction_pri_;
  
  // Compaction分数（用于决定哪个level需要compaction）
  std::vector<double> compaction_score_;
  std::vector<int> compaction_level_;
  
  // 每个level的下一个要compact的文件索引
  std::vector<size_t> next_file_to_compact_by_size_;
  
  // 正在进行的compaction跟踪
  // （在CompactionPicker中维护）
};
```

**关键方法**：
- `ComputeCompactionScore()`：计算每个level的compaction分数
- `UpdateFilesByCompactionPri()`：根据compaction优先级对文件排序
- `GetOverlappingInputs()`：获取与指定key范围重叠的文件

### 3. Compaction（Compaction元数据）

**位置**：`db/compaction/compaction.h:83`

```cpp
class Compaction {
  // 输入文件（可能来自多个level）
  std::vector<CompactionInputFiles> inputs_;
  
  // 输出level
  int output_level_;
  int start_level_;  // 起始level（第一个输入level）
  
  // 输入版本
  Version* input_version_;
  ColumnFamilyData* cfd_;
  
  // VersionEdit（记录compaction的变更）
  VersionEdit edit_;
  
  // Compaction原因
  CompactionReason compaction_reason_;
  
  // 状态标志
  bool is_trivial_move_;      // 是否是trivial move（无需合并）
  bool is_full_compaction_;   // 是否是全量compaction
  bool bottommost_level_;     // 是否是最底层
};
```

**CompactionInputFiles结构**：
```cpp
struct CompactionInputFiles {
  int level;
  std::vector<FileMetaData*> files;  // 该level的输入文件列表
  std::vector<AtomicCompactionUnitBoundary> atomic_compaction_unit_boundaries;
};
```

### 4. CompactionJob（Compaction执行器）

**位置**：`db/compaction/compaction_job.h:143`

```cpp
class CompactionJob {
  CompactionState* compact_;           // Compaction状态
  CompactionStatsFull internal_stats_; // 内部统计信息
  int job_id_;                         // Job ID
  
  // 子compaction状态（用于并行compaction）
  std::vector<SubcompactionState> sub_compact_states_;
  
  // 关键方法
  void Prepare();                      // 准备compaction
  Status Run();                        // 执行compaction
  Status Install(bool* compaction_released); // 安装compaction结果
};
```

---

## Compaction触发机制

### 1. 触发时机

Compaction在以下情况下被触发：

#### A. Compaction完成后
**代码路径**：`db/db_impl/db_impl_compaction_flush.cc:1625`

```cpp
// Compaction完成后，检查是否需要新的compaction
InstallSuperVersionAndScheduleWork(cfd, ...);
  -> MaybeScheduleFlushOrCompaction();
```

#### B. 版本变更后
**代码路径**：`db/db_impl/db_impl.cc:445`

```cpp
// 版本变更后（如手动compaction、文件标记等）
EnqueuePendingCompaction(cfd);
MaybeScheduleFlushOrCompaction();
```

#### C. 配置变更后
**代码路径**：`db/db_impl/db_impl.cc:4429`

```cpp
// MutableCFOptions变更后
EnqueuePendingCompaction(cfd);
MaybeScheduleFlushOrCompaction();
```

### 2. MaybeScheduleFlushOrCompaction()

**代码路径**：`db/db_impl/db_impl_compaction_flush.cc:1760`

**流程**：
1. 检查是否需要compaction（调用`NeedsCompaction()`）
2. 如果需要compaction，将CFD加入compaction队列
3. 调度后台compaction线程

**关键代码**：
```cpp
void DBImpl::MaybeScheduleFlushOrCompaction() {
  // 检查compaction需求
  if (NeedsCompaction()) {
    // 将CFD加入compaction队列
    EnqueuePendingCompaction(cfd);
    // 调度后台线程
    ScheduleCompaction(cfd);
  }
}
```

### 3. NeedsCompaction()判断

**代码路径**：`db/compaction/compaction_picker_level.cc:22`

**判断条件**：
```cpp
bool LevelCompactionPicker::NeedsCompaction(
    const VersionStorageInfo* vstorage) const {
  // 1. 检查是否有TTL过期文件
  if (!vstorage->ExpiredTtlFiles().empty()) return true;
  
  // 2. 检查是否有周期性compaction标记的文件
  if (!vstorage->FilesMarkedForPeriodicCompaction().empty()) return true;
  
  // 3. 检查是否有bottommost文件需要compaction
  if (!vstorage->BottommostFilesMarkedForCompaction().empty()) return true;
  
  // 4. 检查是否有手动标记的文件
  if (!vstorage->FilesMarkedForCompaction().empty()) return true;
  
  // 5. 检查compaction分数（score >= 1.0）
  for (int i = 0; i <= vstorage->MaxInputLevel(); i++) {
    if (vstorage->CompactionScore(i) >= 1) {
      return true;
    }
  }
  return false;
}
```

### 4. Compaction分数计算

**代码路径**：`db/version_set.cc:3740`

**Level 0分数计算**：
```cpp
// L0分数 = L0文件数 / level0_file_num_compaction_trigger
score = static_cast<double>(num_sorted_runs) / 
        mutable_cf_options.level0_file_num_compaction_trigger;
```

**Level 1+分数计算**：
```cpp
// L1+分数 = level大小 / MaxBytesForLevel(level)
score = static_cast<double>(level_bytes_no_compacting) / 
        MaxBytesForLevel(level);
```

**分数排序**：
- 分数按降序排序（`compaction_score_[i]`）
- 分数 >= 1.0 表示需要compaction
- 分数越高，优先级越高

---

## Compaction选择流程（PickCompaction）

### 1. BackgroundCompaction()入口

**代码路径**：`db/db_impl/db_impl_compaction_flush.cc:3726`

**主要流程**：
```cpp
Status DBImpl::BackgroundCompaction(...) {
  // 1. 检查是否有prepicked compaction
  if (prepicked_compaction != nullptr) {
    c.reset(prepicked_compaction->compaction);
  }
  
  // 2. 如果没有prepicked，则选择新的compaction
  if (ShouldPickCompaction(...)) {
    cfd = PickCompactionFromQueue(...);  // 从队列中选择CFD
    c.reset(cfd->PickCompaction(...));   // 选择compaction
  }
  
  // 3. 执行compaction
  if (c) {
    CompactionJob job(...);
    job.Prepare();
    job.Run();
    job.Install();
  }
}
```

### 2. PickCompaction()详细流程

**代码路径**：`db/compaction/compaction_picker_level.cc:991`

**LevelCompactionBuilder::PickCompaction()流程**：

#### 阶段1：SetupInitialFiles() - 选择初始文件

```cpp
void LevelCompactionBuilder::SetupInitialFiles() {
  // 1. 按分数从高到低遍历所有level
  for (int i = 0; i < compaction_picker_->NumberLevels() - 1; i++) {
    start_level_score_ = vstorage_->CompactionScore(i);
    start_level_ = vstorage_->CompactionScoreLevel(i);
    
    if (start_level_score_ >= 1) {
      output_level_ = (start_level_ == 0) ? 
                      vstorage_->base_level() : start_level_ + 1;
      
      // 2. 选择要compact的文件
      if (PickFileToCompact()) {
        compaction_reason_ = (start_level_ == 0) ? 
                            CompactionReason::kLevelL0FilesNum :
                            CompactionReason::kLevelMaxLevelSize;
        break;
      }
    }
  }
  
  // 3. 如果没有找到，检查其他类型的compaction
  // - FilesMarkedForCompaction
  // - BottommostFilesMarkedForCompaction
  // - ExpiredTtlFiles
  // - FilesMarkedForPeriodicCompaction
  // - FilesMarkedForForcedBlobGC
}
```

#### 阶段2：PickFileToCompact() - 选择具体文件

**代码路径**：`db/compaction/compaction_picker_level.cc:792`

```cpp
bool LevelCompactionBuilder::PickFileToCompact() {
  // 1. 获取该level的文件列表和优先级排序
  const std::vector<FileMetaData*>& level_files = 
      vstorage_->LevelFiles(start_level_);
  const std::vector<int>& file_scores = 
      vstorage_->FilesByCompactionPri(start_level_);
  
  // 2. 从next_file_to_compact_by_size_开始遍历
  for (cmp_idx = vstorage_->NextCompactionIndex(start_level_);
       cmp_idx < file_scores.size(); cmp_idx++) {
    int index = file_scores[cmp_idx];
    auto* f = level_files[index];
    
    // 3. 跳过正在被compact的文件
    if (f->being_compacted) continue;
    
    // 4. 添加到输入文件列表
    start_level_inputs_.files.push_back(f);
    
    // 5. 扩展输入文件到clean cut（确保key范围完整）
    if (!compaction_picker_->ExpandInputsToCleanCut(...)) {
      continue;
    }
    
    // 6. 检查是否与正在进行的compaction冲突
    if (compaction_picker_->FilesRangeOverlapWithCompaction(...)) {
      start_level_inputs_.clear();
      continue;
    }
    
    break;  // 找到合适的文件
  }
  
  return start_level_inputs_.size() > 0;
}
```

**文件优先级排序**（UpdateFilesByCompactionPri）：
- `kByCompensatedSize`：按补偿后文件大小降序
- `kOldestLargestSeqFirst`：按最大序列号升序
- `kOldestSmallestSeqFirst`：按最小序列号升序
- `kMinOverlappingRatio`：按与下一层重叠比例升序
- `kRoundRobin`：轮询方式

#### 阶段3：SetupOtherL0FilesIfNeeded() - 处理L0重叠文件

**代码路径**：`db/compaction/compaction_picker_level.cc:330`

```cpp
bool LevelCompactionBuilder::SetupOtherL0FilesIfNeeded() {
  if (start_level_ == 0 && output_level_ != 0) {
    // L0文件可能重叠，需要选择所有重叠的L0文件
    return compaction_picker_->GetOverlappingL0Files(
        vstorage_, &start_level_inputs_, output_level_, &parent_index_);
  }
  return true;
}
```

#### 阶段4：SetupOtherInputsIfNeeded() - 选择输出层的重叠文件

**代码路径**：`db/compaction/compaction_picker_level.cc:91`

```cpp
bool LevelCompactionBuilder::SetupOtherInputsIfNeeded() {
  // 1. 获取输出层的重叠文件
  InternalKey smallest, largest;
  compaction_picker_->GetRange(start_level_inputs_, &smallest, &largest);
  
  CompactionInputFiles output_level_inputs;
  output_level_inputs.level = output_level_;
  vstorage_->GetOverlappingInputs(output_level_, &smallest, &largest,
                                  &output_level_inputs.files);
  
  // 2. 扩展输出层文件到clean cut
  if (!output_level_inputs.empty()) {
    compaction_picker_->ExpandInputsToCleanCut(...);
  }
  
  // 3. 获取grandparents（输出层的下一层，用于限制compaction大小）
  compaction_picker_->GetGrandparents(...);
  
  return true;
}
```

#### 阶段5：GetCompaction() - 创建Compaction对象

```cpp
Compaction* LevelCompactionBuilder::GetCompaction() {
  // 创建Compaction对象
  Compaction* c = new Compaction(
      vstorage_, immutable_options_, mutable_cf_options_,
      mutable_db_options_, compaction_inputs_, output_level_,
      target_file_size_, max_compaction_bytes_, ...);
  
  // 注册compaction（标记文件为being_compacted）
  compaction_picker_->RegisterCompaction(c);
  
  return c;
}
```

### 3. 关键辅助函数

#### ExpandInputsToCleanCut() - 扩展输入到完整key范围

**目的**：确保compaction包含完整的key范围，避免遗漏数据

```cpp
bool CompactionPicker::ExpandInputsToCleanCut(...) {
  // 1. 向左扩展：添加与当前文件key范围重叠的文件
  // 2. 向右扩展：添加与当前文件key范围重叠的文件
  // 3. 确保所有文件形成一个连续的key范围
}
```

#### GetOverlappingInputs() - 获取重叠文件

**目的**：找到与指定key范围重叠的所有文件

```cpp
void VersionStorageInfo::GetOverlappingInputs(
    int level, const InternalKey* begin, const InternalKey* end,
    std::vector<FileMetaData*>* inputs) {
  // 使用二分查找找到重叠的文件
  // 对于L0，需要检查所有文件（可能重叠）
  // 对于L1+，文件不重叠，只需检查边界文件
}
```

---

## Compaction执行流程（CompactionJob）

### 1. CompactionJob::Prepare() - 准备阶段

**代码路径**：`db/compaction/compaction_job.cc:264`

**主要工作**：
```cpp
void CompactionJob::Prepare(...) {
  // 1. 生成子compaction边界（用于并行compaction）
  if (c->ShouldFormSubcompactions()) {
    GenSubcompactionBoundaries();
    // 为每个子compaction创建SubcompactionState
    for (size_t i = 0; i <= boundaries_.size(); i++) {
      compact_->sub_compact_states.emplace_back(...);
    }
  } else {
    // 单个compaction
    compact_->sub_compact_states.emplace_back(...);
  }
  
  // 2. 收集seqno->time映射信息（用于时间相关的compaction）
  // 3. 计算preserve_seqno_after_和proximal_after_seqno_
  // 4. 设置compaction filter
}
```

**子compaction划分**：
- 根据key范围将compaction划分为多个子任务
- 每个子compaction可以并行执行
- 边界通过采样输入文件确定

### 2. CompactionJob::Run() - 执行阶段

**代码路径**：`db/compaction/compaction_job.cc:1068`

**主要流程**：
```cpp
Status CompactionJob::Run() {
  InitializeCompactionRun();
  
  // 1. 执行子compaction（并行）
  RunSubcompactions();
  
  // 2. 收集子compaction错误
  Status status = CollectSubcompactionErrors();
  
  // 3. 同步输出目录
  if (status.ok()) {
    status = SyncOutputDirectories();
  }
  
  // 4. 验证输出文件
  if (status.ok()) {
    status = VerifyOutputFiles();
  }
  
  // 5. 设置输出文件属性
  if (status.ok()) {
    SetOutputTableProperties();
  }
  
  // 6. 聚合统计信息
  AggregateSubcompactionOutputAndJobStats();
  
  // 7. 更新内部统计
  UpdateInternalStatsFromInputFiles(...);
  
  // 8. 验证compaction记录数
  if (status.ok()) {
    status = VerifyCompactionRecordCounts(...);
  }
  
  // 9. 完成compaction运行
  FinalizeCompactionRun(status, ...);
  
  return status;
}
```

### 3. RunSubcompactions() - 并行执行

**代码路径**：`db/compaction/compaction_job.cc:718`

```cpp
void CompactionJob::RunSubcompactions() {
  const size_t num_threads = compact_->sub_compact_states.size();
  
  // 1. 为子compaction 1..N-1 启动线程
  std::vector<port::Thread> thread_pool;
  for (size_t i = 1; i < num_threads; i++) {
    thread_pool.emplace_back(&CompactionJob::ProcessKeyValueCompaction, 
                             this, &compact_->sub_compact_states[i]);
  }
  
  // 2. 在当前线程执行第一个子compaction
  ProcessKeyValueCompaction(&compact_->sub_compact_states[0]);
  
  // 3. 等待所有线程完成
  for (auto& thread : thread_pool) {
    thread.join();
  }
  
  // 4. 清理空输出
  RemoveEmptyOutputs();
  ReleaseSubcompactionResources();
}
```

### 4. ProcessKeyValueCompaction() - 处理KV compaction

**核心流程**：
```cpp
void CompactionJob::ProcessKeyValueCompaction(SubcompactionState* sub_compact) {
  // 1. 创建CompactionIterator（合并多个输入文件的迭代器）
  std::unique_ptr<InternalIterator> input(NewInternalIterator(...));
  
  // 2. 创建CompactionIterator（处理合并、删除、过滤等）
  CompactionIterator c_iter(input.get(), ...);
  
  // 3. 创建TableBuilder（构建输出SST文件）
  std::unique_ptr<TableBuilder> builder;
  
  // 4. 迭代处理每个key-value
  for (c_iter.SeekToFirst(); c_iter.Valid(); c_iter.Next()) {
    // 4.1 检查是否需要切换输出文件
    if (ShouldSwitchOutputFile(sub_compact, c_iter)) {
      FinishCompactionOutputFile(sub_compact, ...);
      OpenCompactionOutputFile(sub_compact, ...);
    }
    
    // 4.2 添加key-value到当前输出文件
    builder->Add(c_iter.key(), c_iter.value());
    
    // 4.3 更新统计信息
    sub_compact->compaction_job_stats.num_input_records++;
  }
  
  // 5. 完成最后一个输出文件
  FinishCompactionOutputFile(sub_compact, ...);
}
```

**CompactionIterator的作用**：
- 合并多个输入文件的key-value
- 处理相同key的多个版本（保留最新的）
- 应用compaction filter（删除过期数据）
- 处理范围删除（range deletion）

### 5. OpenCompactionOutputFile() - 打开输出文件

**代码路径**：`db/compaction/compaction_job.cc:2460`

```cpp
Status CompactionJob::OpenCompactionOutputFile(...) {
  // 1. 分配新的文件号
  uint64_t file_number = versions_->NewFileNumber();
  
  // 2. 构建文件路径
  std::string fname = GetTableFileName(file_number);
  
  // 3. 创建WritableFile
  std::unique_ptr<FSWritableFile> writable_file;
  IOStatus io_s = NewWritableFile(fs_.get(), fname, &writable_file, fo_copy);
  
  // 4. 创建TableBuilder
  std::unique_ptr<TableBuilder> builder(
      table_factory_->NewTableBuilder(
          TableBuilderOptions(...), 
          writable_file.get()));
  
  // 5. 记录输出文件信息
  outputs.AddOutputFile(file_number, fname, ...);
}
```

### 6. FinishCompactionOutputFile() - 完成输出文件

**代码路径**：`db/compaction/compaction_job.cc:2800+`

```cpp
Status CompactionJob::FinishCompactionOutputFile(...) {
  // 1. 完成TableBuilder（写入footer等）
  Status s = builder->Finish();
  
  // 2. 同步文件
  if (s.ok()) {
    s = output_file_writer->Sync(...);
  }
  
  // 3. 关闭文件
  if (s.ok()) {
    s = output_file_writer->Close(...);
  }
  
  // 4. 创建FileMetaData
  FileMetaData meta;
  meta.fd = FileDescriptor(file_number, path_id, file_size);
  meta.smallest = smallest_key;
  meta.largest = largest_key;
  // ... 其他属性
  
  // 5. 添加到输出文件列表
  outputs.AddOutputFile(meta);
  
  // 6. 如果是Phase2模式，进行ML预测和二次写入
  if (enable_phase2 && output_level > 0) {
    // 收集特征、预测、映射handle、二次写入
    CollectFeaturesAndPredict(...);
    RewriteFileToTargetHandle(...);
  }
}
```

### 7. CompactionJob::Install() - 安装结果

**代码路径**：`db/compaction/compaction_job.cc:1119`

```cpp
Status CompactionJob::Install(bool* compaction_released) {
  // 1. 更新内部统计
  cfd->internal_stats()->AddCompactionStats(output_level, ...);
  
  // 2. 安装compaction结果
  if (status.ok()) {
    status = InstallCompactionResults(compaction_released);
  }
  
  return status;
}
```

### 8. InstallCompactionResults() - 安装compaction结果

**代码路径**：`db/compaction/compaction_job.cc:2341`

```cpp
Status CompactionJob::InstallCompactionResults(bool* compaction_released) {
  VersionEdit* edit = compaction->edit();
  
  // 1. 添加输入文件删除
  compaction->AddInputDeletions(edit);
  
  // 2. 添加输出文件
  for (const auto& sub_compact : compact_->sub_compact_states) {
    sub_compact.AddOutputsEdit(edit);
  }
  
  // 3. 应用VersionEdit到VersionSet（LogAndApply）
  return versions_->LogAndApply(
      compaction->column_family_data(), read_options, write_options,
      edit, db_mutex_, db_directory_, ...,
      [&compaction, &compaction_released](const Status& s) {
        compaction->ReleaseCompactionFiles(s);
        *compaction_released = true;
      });
}
```

**LogAndApply的作用**：
- 将VersionEdit写入MANIFEST文件
- 创建新的Version对象
- 更新VersionSet的current版本
- 触发SuperVersion的安装

---

## 状态变化与生命周期

### 1. 文件状态变化

```
FileMetaData状态转换：

创建阶段：
  FileMetaData创建
    -> being_compacted = false
    -> refs = 1

Compaction选择阶段：
  PickCompaction()
    -> RegisterCompaction(c)
      -> 对每个输入文件：being_compacted = true

Compaction执行阶段：
  CompactionJob::Run()
    -> 读取输入文件
    -> 写入输出文件

Compaction完成阶段：
  InstallCompactionResults()
    -> LogAndApply()
      -> ReleaseCompactionFiles()
        -> 对每个输入文件：being_compacted = false
        -> 输入文件refs--（可能被删除）
        -> 输出文件refs++（新Version引用）
```

### 2. Compaction对象生命周期

```
Compaction对象生命周期：

1. 创建：
   PickCompaction()
     -> new Compaction(...)
     -> MarkFilesBeingCompacted(true)

2. 执行：
   CompactionJob job(compaction, ...)
   job.Prepare()
   job.Run()
   job.Install()

3. 释放：
   InstallCompactionResults()
     -> ReleaseCompactionFiles()
       -> MarkFilesBeingCompacted(false)
       -> delete Compaction
```

### 3. Version变更流程

```
Version变更流程：

1. 当前Version（V1）：
   - files_[level] = [f1, f2, f3, ...]
   - 被多个迭代器和compaction引用

2. Compaction执行：
   - 读取V1中的文件f1, f2
   - 生成新文件f10, f11

3. VersionEdit创建：
   - DeleteFile(level, f1)
   - DeleteFile(level, f2)
   - AddFile(level, f10)
   - AddFile(level, f11)

4. LogAndApply：
   - 写入MANIFEST
   - 创建新Version（V2）
   - V2.files_[level] = [f3, f10, f11, ...]
   - V1.refs--（如果refs==0，V1被删除）

5. SuperVersion安装：
   - 更新current_指向V2
   - 旧的SuperVersion被标记为obsolete
   - 等待所有引用释放后删除
```

### 4. CompactionPicker状态跟踪

```cpp
class CompactionPicker {
  // 正在进行的L0 compaction
  std::set<Compaction*> level0_compactions_in_progress_;
  
  // 正在进行的其他level compaction
  std::unordered_set<Compaction*> compactions_in_progress_;
  
  void RegisterCompaction(Compaction* c) {
    if (c->start_level() == 0) {
      level0_compactions_in_progress_.insert(c);
    }
    compactions_in_progress_.insert(c);
  }
  
  void UnregisterCompaction(Compaction* c) {
    if (c->start_level() == 0) {
      level0_compactions_in_progress_.erase(c);
    }
    compactions_in_progress_.erase(c);
  }
};
```

---

## 关键代码路径

### 1. Compaction触发路径

```
MaybeScheduleFlushOrCompaction()  // 或直接调用
  -> NeedsCompaction()
    -> VersionStorageInfo::ComputeCompactionScore()
      -> 计算每个level的分数
  -> EnqueuePendingCompaction(cfd)
  -> ScheduleCompaction(cfd)
    -> BGWorkCompaction()
      -> BackgroundCompaction()
```

### 2. Compaction选择路径

```
BackgroundCompaction()
  -> PickCompactionFromQueue()
    -> 从队列中选择CFD
  -> cfd->PickCompaction()
    -> LevelCompactionPicker::PickCompaction()
      -> LevelCompactionBuilder::PickCompaction()
        -> SetupInitialFiles()
          -> PickFileToCompact()
        -> SetupOtherL0FilesIfNeeded()
        -> SetupOtherInputsIfNeeded()
        -> GetCompaction()
          -> new Compaction(...)
          -> RegisterCompaction(c)
```

### 3. Compaction执行路径

```
BackgroundCompaction()
  -> CompactionJob job(compaction, ...)
  -> job.Prepare()
    -> GenSubcompactionBoundaries()
    -> 创建SubcompactionState
  -> job.Run()
    -> RunSubcompactions()
      -> ProcessKeyValueCompaction()
        -> 创建CompactionIterator
        -> 迭代处理key-value
        -> OpenCompactionOutputFile()
        -> FinishCompactionOutputFile()
  -> job.Install()
    -> InstallCompactionResults()
      -> versions_->LogAndApply()
        -> 写入MANIFEST
        -> 创建新Version
        -> ReleaseCompactionFiles()
```

### 4. Version变更路径

```
LogAndApply()
  -> VersionBuilder::Apply(edit)
    -> 应用VersionEdit到VersionStorageInfo
  -> VersionBuilder::SaveTo(vstorage)
    -> 构建新的VersionStorageInfo
  -> new Version(cfd, vstorage)
  -> AppendVersion(v)
    -> current_ = v
    -> obsolete_versions_.push_back(old_current)
  -> InstallSuperVersionAndScheduleWork()
    -> 安装新的SuperVersion
    -> 调度新的compaction
```

---

## 总结

RocksDB的Compaction是一个复杂的多阶段过程：

1. **触发阶段**：通过计算compaction分数判断是否需要compaction
2. **选择阶段**：根据优先级选择文件和level，构建Compaction对象
3. **执行阶段**：并行执行子compaction，合并文件，应用过滤
4. **安装阶段**：通过VersionEdit更新VersionSet，安装新Version

整个过程涉及多个数据结构的协调：
- **FileMetaData**：跟踪文件状态和元数据
- **VersionStorageInfo**：管理每个level的文件列表和compaction分数
- **Compaction**：描述一次compaction的输入输出
- **CompactionJob**：执行compaction的具体工作
- **Version/VersionSet**：管理LSM-Tree的版本历史

理解这些流程对于实现自定义compaction策略（如基于ML预测的compaction）至关重要。

