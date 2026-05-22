# Compaction File Selection Scheme Based on Predicted Deletion Time

## 设计概述

基于预测删除时间的 Compaction 文件选择方案，通过优先队列（最小堆）管理文件，总是选择剩余生命周期最短的文件进行 Compaction。

## 数据结构

### 1. FileLifetimeInfo
```cpp
struct FileLifetimeInfo {
  uint64_t file_number;              // 文件号
  uint64_t predicted_deletion_time;  // 预测删除时间（微秒）
  double predicted_lifetime;        // 预测的生命周期（秒）
  uint64_t creation_time;           // 文件创建时间（微秒）
  uint64_t file_size;               // 文件大小（字节）
  double remaining_lifetime;        // 剩余生命周期（秒）= predicted_deletion_time - current_time
  int level;                         // 文件所在的 level
  
  // 用于优先队列比较
  bool operator>(const FileLifetimeInfo& other) const {
    return remaining_lifetime > other.remaining_lifetime;
  }
};
```

### 2. 优先队列
使用 `std::priority_queue` 实现最小堆，按 `remaining_lifetime` 排序。

## 实现步骤

### Phase 1: 文件写入阶段

1. **文件写入临时目录**
   - 文件写入 `db_path/../../tmp/xxx.sst`
   - 已在 `TwoPhaseWriteManager::HandleFileCreation` 中实现

2. **特征收集和预测**
   - 在 `CollectFeaturesAndPredict` 中：
     - 收集特征
     - 使用 ML 模型预测生命周期
     - 计算预测删除时间 = 当前时间 + 预测生命周期
     - 创建 `FileLifetimeInfo` 对象
     - 计算剩余生命周期 = 预测删除时间 - 当前时间
     - 插入优先队列

3. **文件重写**
   - 从临时目录移动到目标目录，使用对应的 handle
   - 已在 `RewriteFileToTargetHandle` 中实现

### Phase 2: Compaction 触发阶段

1. **CustomCompactionPicker**
   - 继承 `CompactionPicker` 或 `LevelCompactionPicker`
   - 重写 `PickCompaction` 方法

2. **文件选择逻辑**
   - 从优先队列顶部获取文件（剩余生命周期最短）
   - 检查文件状态（是否存在、是否有效）
   - 检查文件数量限制（每次最多 1-2 个文件，可配置）
   - 如果还需要更多文件，继续从优先队列获取
   - 创建 `Compaction` 对象并返回

3. **文件状态更新**
   - 当文件被选中进行 Compaction 时，从优先队列中移除
   - Compaction 完成后，文件被删除或合并

## 接口设计

### TwoPhaseWriteManager 新增接口

```cpp
// 获取下一个应该被 Compaction 的文件（剩余生命周期最短）
bool GetNextFileForCompaction(FileLifetimeInfo* file_info);

// 从优先队列中移除文件（当文件被选中进行 Compaction 时）
void RemoveFileFromQueue(uint64_t file_number);

// 更新文件的剩余生命周期（定期刷新）
void RefreshRemainingLifetimes();
```

### CustomCompactionPicker

```cpp
class CustomCompactionPicker : public LevelCompactionPicker {
 public:
  CustomCompactionPicker(const ImmutableOptions& ioptions,
                         const InternalKeyComparator* icmp);
  
  Compaction* PickCompaction(...) override;
  
 private:
  // 从优先队列选择文件
  std::vector<FileMetaData*> SelectFilesFromPriorityQueue(
      VersionStorageInfo* vstorage, int max_files);
};
```

## 配置选项

- `max_files_per_compaction`: 每次 Compaction 最多选择的文件数（默认 1-2）
- `enable_predicted_compaction`: 是否启用基于预测的 Compaction 选择

## 注意事项

1. **线程安全**：优先队列需要线程安全保护（使用 mutex）
2. **文件状态检查**：需要验证文件是否仍然存在且有效
3. **定期刷新**：剩余生命周期需要定期更新（因为时间在流逝）
4. **Level 0 处理**：Level 0 文件可能不使用此策略（需要确认）









