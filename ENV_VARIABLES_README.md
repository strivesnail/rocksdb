# RocksDB 环境变量控制说明

## 环境变量列表

### 1. ROCKSDB_ENABLE_PHASE2
**控制是否启用二次写入（Phase 2模式）**

- **启用**: `export ROCKSDB_ENABLE_PHASE2=1`
- **禁用**: `unset ROCKSDB_ENABLE_PHASE2` 或 `export ROCKSDB_ENABLE_PHASE2=0`
- **默认**: 禁用（Phase 1模式）

**功能说明**:
- Phase 1模式（禁用）: 文件直接写入目标目录，不进行预测和二次写入
- Phase 2模式（启用）: 文件先写入tmp目录，收集特征并预测生命周期，然后移动到目标handle

### 2. ROCKSDB_ML_MODELS_PATH
**ML模型目录路径**

- **设置**: `export ROCKSDB_ML_MODELS_PATH=/path/to/models`
- **默认**: `/home/usr/test_environment/workplace/log/experiment_logs/new_log/run_batch_20260131_063807/backup_old_data_models/models_balanced`

**功能说明**: 指定ML模型文件所在的目录，包含按level训练的模型文件。

### 3. ROCKSDB_TOOLS_PATH
**Python工具脚本路径（可选）**

- **设置**: `export ROCKSDB_TOOLS_PATH=/path/to/tools`
- **默认**: `./tools`（相对于当前工作目录）

**功能说明**: 指定 `ml_predict_lifetime_by_level.py` 脚本所在的目录。

## 使用方式

### 方式1: 直接在命令行设置

```bash
# 启用二次写入
export ROCKSDB_ENABLE_PHASE2=1
export ROCKSDB_ML_MODELS_PATH=/path/to/models
./your_rocksdb_program

# 禁用二次写入
unset ROCKSDB_ENABLE_PHASE2
./your_rocksdb_program
```

### 方式2: 使用提供的Shell脚本

```bash
# 启用二次写入
./run_with_phase2.sh ./db_bench --benchmarks=fillrandom

# 禁用二次写入
./run_without_phase2.sh ./db_bench --benchmarks=fillrandom
```

### 方式3: 在Shell脚本中设置

创建你自己的脚本：

```bash
#!/bin/bash
# 设置环境变量
export ROCKSDB_ENABLE_PHASE2=1
export ROCKSDB_ML_MODELS_PATH="/path/to/models"

# 执行程序
./your_rocksdb_program "$@"
```

### 方式4: 在Makefile或CMake中设置

```makefile
# Makefile示例
run-phase2:
	ROCKSDB_ENABLE_PHASE2=1 \
	ROCKSDB_ML_MODELS_PATH=/path/to/models \
	./your_program
```

## 验证环境变量

在程序中，环境变量的读取逻辑在 `db/db_impl/db_impl_open.cc` 中：

```cpp
// 读取 enable_phase2
const char* enable_phase2_env = std::getenv("ROCKSDB_ENABLE_PHASE2");
bool enable_phase2 = (enable_phase2_env != nullptr && 
                     std::string(enable_phase2_env) == "1");
```

## 注意事项

1. **环境变量必须在程序启动前设置**，程序启动后读取一次，运行时不会重新读取
2. **ROCKSDB_ENABLE_PHASE2** 必须精确设置为 `"1"` 才会启用，其他任何值（包括 `"true"`, `"yes"` 等）都会被当作禁用
3. 如果模型路径不存在或模型加载失败，程序会记录警告但不会阻止数据库打开
4. Phase 2模式需要Python环境和ML模型文件，如果缺少这些，二次写入功能可能无法正常工作

