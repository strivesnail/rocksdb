# Compaction File Selection Scheme Based on Predicted Deletion Time (V2)

## 设计概述

基于预测删除时间的 Compaction 文件选择方案，通过优先队列（最小堆）管理文件，在特定条件下选择"离预测生命周期太远"的文件进行 Compaction。

## 关键约束

1. **Level 0**: 永远写入 handle6 (WLTH_LEVEL0)，不做任何特殊操作
2. **Phase 1**: 不做任何修改，保持原样
3. **Phase 2**: 实现自定义 compaction 函数，但只在特定条件下启用

## Handle 生命周期映射

根据 `MapLifetimeToHandle` 函数：
- **Handle 6**: [0, 8) seconds
- **Handle 7**: [8, 12) seconds
- **Handle 8**: [12, 30) seconds
- **Handle 9**: [30, 100) seconds
- **Handle 10**: [100, 300) seconds
- **Handle 11**: [300, 600) seconds
- **Handle 12**: [600, +∞) seconds

## "太远"的判断逻辑

### 规则 1: 非最后一个 Handle（6-11）

如果文件已经存活的时间 > **下一个 handle 的生命周期上限**，则认为"太远"。

**示例：**
- 文件在 Handle 7（生命周期 [8, 12)s），Handle 8 的上限是 30s
  - 如果文件存活时间 > 30s，则选它
- 文件在 Handle 8（生命周期 [12, 30)s），Handle 9 的上限是 100s
  - 如果文件存活时间 > 100s，则选它
- 文件在 Handle 11（生命周期 [300, 600)s），Handle 12 的下限是 600s
  - 如果文件存活时间 > 600s，则选它

### 规则 2: Handle 11（倒数第二个）

对于 Handle 11，使用特殊逻辑：
- Handle 11: [300, 600)s
- Handle 12 的下限: 600s
- 如果文件存活时间 > **Handle 12 下限的 1.5 倍**（即 600 * 1.5 = 900s），则选它

**示例：**
- Handle 11 的文件：存活时间 > 900s → 选它

### 规则 3: Handle 12（最后一个）

**Handle 12 的文件不需要 Compaction，因此：**
- Handle 12 的文件不参与"太远"判断
- Handle 12 的文件不进入优先队列（或者进入但不参与选择）
- Handle 12 的文件不参与 Compaction 计数

## 数据结构

### FileLifetimeInfo
```cpp
struct FileLifetimeInfo {
  uint64_t file_number;              // 文件号
  uint64_t predicted_deletion_time;  // 预测删除时间（微秒）
  double predicted_lifetime;         // 预测的生命周期（秒）
  uint64_t creation_time;            // 文件创建时间（微秒）
  uint64_t file_size;                // 文件大小（字节）
  int handle;                        // 文件所在的 handle
  int level;                         // 文件所在的 level
  
  // 计算当前存活时间（秒）
  double GetCurrentAge() const {
    uint64_t current_time = Env::Default()->NowMicros();
    return (current_time - creation_time) / 1000000.0;
  }
  
  // 判断是否"太远"
  bool IsTooFar() const {
    double current_age = GetCurrentAge();
    return IsTooFarForHandle(handle, current_age);
  }
  
  // 根据 handle 判断是否太远
  static bool IsTooFarForHandle(int handle, double current_age_seconds);
};
```

## 工作流程

### Phase 2: 文件写入阶段

1. 文件写入临时目录 `db_path/../../tmp/xxx.sst`
2. 特征收集和预测
3. 根据预测生命周期映射到对应的 handle
4. 文件重写到目标目录，使用对应的 handle
5. **创建 FileLifetimeInfo 并插入优先队列**（按剩余生命周期排序）

### Phase 2: Compaction 触发阶段

1. **维护 Compaction 计数器**
   - 对每个 level（不包括 level 0）维护一个计数器
   - 记录该 level 的 normal compaction 次数

2. **Compaction 选择逻辑**
   - 每做 4 次正常 compaction，第 5 次检查：
     - 遍历优先队列，查找是否有文件"太远"
     - 如果有，选择"最远"的文件（存活时间最长的）
     - 如果没有，使用正常的 RocksDB compaction 选择逻辑

3. **文件删除时的清理**
   - 当 RocksDB 删除文件时，从优先队列中移除对应的 FileLifetimeInfo

## 实现细节

### 1. Handle 生命周期范围定义

```cpp
struct HandleLifetimeRange {
  int handle;
  double lower_bound;  // 下限（秒）
  double upper_bound;  // 上限（秒），+∞ 用 -1 表示
};

static const HandleLifetimeRange HANDLE_RANGES[] = {
  {6, 0.0, 8.0},
  {7, 8.0, 12.0},
  {8, 12.0, 30.0},
  {9, 30.0, 100.0},
  {10, 100.0, 300.0},
  {11, 300.0, 600.0},
  {12, 600.0, -1.0}  // -1 表示 +∞
};
```

### 2. "太远"判断函数

```cpp
bool FileLifetimeInfo::IsTooFarForHandle(int handle, double current_age_seconds) {
  // Handle 12 不需要 Compaction，直接返回 false
  if (handle == 12) {
    return false;
  }
  
  // Handle 11: 如果存活时间 > Handle 12 下限的 1.5 倍（900s）
  if (handle == 11) {
    HandleLifetimeRange handle12_range = GetHandleRange(12);
    return current_age_seconds > (handle12_range.lower_bound * 1.5);  // 600 * 1.5 = 900
  }
  
  // Handle 6-10: 如果存活时间 > 下一个 handle 的上限
  int next_handle = handle + 1;
  HandleLifetimeRange next_range = GetHandleRange(next_handle);
  
  // 如果下一个 handle 的上限是 -1（Handle 12），不应该到这里（因为 handle 11 已经处理了）
  if (next_range.upper_bound < 0) {
    return false;
  }
  
  // 普通情况：如果存活时间 > 下一个 handle 的上限
  return current_age_seconds > next_range.upper_bound;
}
```

### 3. 优先队列管理

```cpp
class TwoPhaseWriteManager {
  // 优先队列：按剩余生命周期排序（最小堆）
  // 注意：这里需要按"存活时间"排序，而不是"剩余生命周期"
  // 因为我们要找"最远"的文件，即存活时间最长的
  std::priority_queue<FileLifetimeInfo, 
                      std::vector<FileLifetimeInfo>,
                      std::function<bool(const FileLifetimeInfo&, const FileLifetimeInfo&)>> 
      file_lifetime_queue_;
  
  // 文件号到 FileLifetimeInfo 的映射（用于快速查找和删除）
  std::unordered_map<uint64_t, FileLifetimeInfo> file_lifetime_map_;
  
  // 每个 level 的 compaction 计数器（不包括 level 0）
  std::unordered_map<int, int> compaction_counters_;
  
  // 线程安全保护
  std::shared_mutex queue_mutex_;
};
```

### 4. Compaction 选择逻辑

```cpp
// 在 CustomCompactionPicker::PickCompaction 中
Compaction* CustomCompactionPicker::PickCompaction(...) {
  // 获取当前 level
  int level = ...;
  
  // 如果是 level 0，使用正常逻辑
  if (level == 0) {
    return LevelCompactionPicker::PickCompaction(...);
  }
  
  // 检查计数器
  int& counter = compaction_counters_[level];
  counter++;
  
  // 每 5 次检查一次（第 1, 2, 3, 4 次用正常逻辑，第 5 次检查）
  if (counter % 5 != 0) {
    return LevelCompactionPicker::PickCompaction(...);
  }
  
  // 第 5 次：检查是否有"太远"的文件
  FileLifetimeInfo too_far_file;
  if (g_two_phase_write_manager->FindTooFarFile(level, &too_far_file)) {
    // 找到"太远"的文件，使用自定义选择逻辑
    return SelectFileForCompaction(too_far_file, vstorage, ...);
  }
  
  // 没有"太远"的文件，使用正常逻辑
  return LevelCompactionPicker::PickCompaction(...);
}
```

## 环境变量控制

- `ROCKSDB_ENABLE_PREDICTED_COMPACTION=1`: 启用基于预测的 Compaction 选择
- 即使设置了环境变量，也只在特定条件下使用（每 5 次中的第 5 次）

## 关键修正

1. **Handle 12 的处理**：
   - ✅ Handle 12 的文件**不需要 Compaction**，不参与"太远"判断
   - ✅ Handle 12 的文件不进入优先队列（或进入但不参与选择）
   - ✅ Handle 12 的文件不参与 Compaction 计数

2. **Handle 11 的判断阈值**：
   - ✅ 使用 **900s**（600 * 1.5），而不是 600s
   - ✅ 如果 Handle 11 的文件存活时间 > 900s，则选它

## 待讨论的问题

1. **"最远"的定义**：
   - 当前设计：选择存活时间最长的文件
   - 是否需要考虑其他因素（如文件大小、level 等）？

2. **计数器重置**：
   - 计数器是否需要定期重置？
   - 还是持续累加？

3. **多文件选择**：
   - 如果找到多个"太远"的文件，是否都选择？
   - 还是只选择一个最远的？

4. **Handle 12 文件的处理**：
   - Handle 12 的文件是否完全不进入优先队列？
   - 还是进入但不参与选择？

