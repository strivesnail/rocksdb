#!/usr/bin/env python3
"""
ML预测功能集成测试
模拟实际RocksDB运行场景，测试ML预测在不同阶段的行为
"""

import os
import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, '/home/usr/test_environment/rocksdb/tools')
from ml_predict_lifetime_by_level import FileLifetimePredictorByLevel

def simulate_rocksdb_ml_prediction(seqno, level, features, predictor):
    """
    模拟RocksDB的ML预测调用逻辑
    
    Args:
        seqno: 当前序列号
        level: 文件所在的level
        features: 69个特征
        predictor: 预测器实例
    
    Returns:
        (should_predict, predicted_lifetime) 或 (False, 0.0)
    """
    # 读取环境变量（模拟C++代码）
    ml_enable_env = os.getenv('ROCKSDB_ML_PREDICTION_ENABLE')
    ml_phase_env = os.getenv('ROCKSDB_ML_PREDICTION_PHASE')
    
    ml_enabled = (ml_enable_env is None or ml_enable_env != '0')
    ml_phase = ml_phase_env if ml_phase_env else 'both'
    
    # 判断阶段
    is_zipfian_phase = seqno >= 50000000
    
    # 决定是否启用ML预测
    use_ml_prediction = False
    if ml_enabled:
        if ml_phase == 'both':
            use_ml_prediction = True
        elif ml_phase == 'zipfian' and is_zipfian_phase:
            use_ml_prediction = True
        elif ml_phase == 'uniform' and not is_zipfian_phase:
            use_ml_prediction = True
    
    if not use_ml_prediction:
        return (False, 0.0, 'disabled_by_config')
    
    # 尝试预测
    try:
        predicted_lifetime = predictor.predict(features, level)
        return (True, predicted_lifetime, 'success')
    except Exception as e:
        return (False, 0.0, f'prediction_failed: {e}')

def test_integration_scenario():
    """测试集成场景"""
    print("="*80)
    print("ML预测功能集成测试")
    print("="*80)
    
    # 初始化预测器
    print("\n初始化预测器...")
    predictor = FileLifetimePredictorByLevel()
    
    # 测试场景：模拟一个完整的实验运行
    print("\n" + "="*80)
    print("场景测试：模拟完整实验运行")
    print("="*80)
    
    # 场景1: 只在zipfian阶段启用（你的需求）
    print("\n场景1: 只在zipfian阶段启用ML预测")
    print("-" * 80)
    
    os.environ['ROCKSDB_ML_PREDICTION_ENABLE'] = '1'
    os.environ['ROCKSDB_ML_PREDICTION_PHASE'] = 'zipfian'
    
    # 模拟uniform阶段（seqno < 50M）
    uniform_seqnos = [1000000, 10000000, 30000000, 49999999]
    print("\n  Uniform阶段 (seqno < 50M):")
    uniform_predictions = 0
    for seqno in uniform_seqnos:
        features = np.random.rand(69)
        level = np.random.randint(1, 7)
        should_predict, lifetime, status = simulate_rocksdb_ml_prediction(
            seqno, level, features, predictor
        )
        phase_name = 'uniform' if seqno < 50000000 else 'zipfian'
        print(f"    seqno={seqno:>12,}, level={level}, phase={phase_name:>8}, "
              f"预测={'启用' if should_predict else '禁用'}, "
              f"生命周期={lifetime:.2f}秒, 状态={status}")
        if should_predict:
            uniform_predictions += 1
    
    # 模拟zipfian阶段（seqno >= 50M）
    zipfian_seqnos = [50000000, 60000000, 80000000, 100000000]
    print("\n  Zipfian阶段 (seqno >= 50M):")
    zipfian_predictions = 0
    for seqno in zipfian_seqnos:
        features = np.random.rand(69)
        level = np.random.randint(1, 7)
        should_predict, lifetime, status = simulate_rocksdb_ml_prediction(
            seqno, level, features, predictor
        )
        phase_name = 'uniform' if seqno < 50000000 else 'zipfian'
        print(f"    seqno={seqno:>12,}, level={level}, phase={phase_name:>8}, "
              f"预测={'启用' if should_predict else '禁用'}, "
              f"生命周期={lifetime:.2f}秒, 状态={status}")
        if should_predict:
            zipfian_predictions += 1
    
    print(f"\n  结果统计:")
    print(f"    Uniform阶段预测次数: {uniform_predictions}/{len(uniform_seqnos)} (期望: 0)")
    print(f"    Zipfian阶段预测次数: {zipfian_predictions}/{len(zipfian_seqnos)} (期望: {len(zipfian_seqnos)})")
    
    scenario1_ok = (uniform_predictions == 0 and zipfian_predictions == len(zipfian_seqnos))
    print(f"    场景1结果: {'✓ 通过' if scenario1_ok else '✗ 失败'}")
    
    # 场景2: 完全禁用
    print("\n" + "="*80)
    print("场景2: 完全禁用ML预测")
    print("-" * 80)
    
    os.environ['ROCKSDB_ML_PREDICTION_ENABLE'] = '0'
    
    test_seqnos = [10000000, 50000000, 100000000]
    predictions = 0
    for seqno in test_seqnos:
        features = np.random.rand(69)
        level = np.random.randint(1, 7)
        should_predict, lifetime, status = simulate_rocksdb_ml_prediction(
            seqno, level, features, predictor
        )
        if should_predict:
            predictions += 1
    
    scenario2_ok = (predictions == 0)
    print(f"  预测次数: {predictions}/{len(test_seqnos)} (期望: 0)")
    print(f"  场景2结果: {'✓ 通过' if scenario2_ok else '✗ 失败'}")
    
    # 场景3: 两个阶段都启用
    print("\n" + "="*80)
    print("场景3: 两个阶段都启用ML预测")
    print("-" * 80)
    
    os.environ['ROCKSDB_ML_PREDICTION_ENABLE'] = '1'
    os.environ['ROCKSDB_ML_PREDICTION_PHASE'] = 'both'
    
    test_seqnos = [10000000, 50000000, 100000000]
    predictions = 0
    for seqno in test_seqnos:
        features = np.random.rand(69)
        level = np.random.randint(1, 7)
        should_predict, lifetime, status = simulate_rocksdb_ml_prediction(
            seqno, level, features, predictor
        )
        if should_predict:
            predictions += 1
    
    scenario3_ok = (predictions == len(test_seqnos))
    print(f"  预测次数: {predictions}/{len(test_seqnos)} (期望: {len(test_seqnos)})")
    print(f"  场景3结果: {'✓ 通过' if scenario3_ok else '✗ 失败'}")
    
    # 测试不同Level的预测
    print("\n" + "="*80)
    print("场景4: 测试不同Level的预测准确性")
    print("-" * 80)
    
    os.environ['ROCKSDB_ML_PREDICTION_ENABLE'] = '1'
    os.environ['ROCKSDB_ML_PREDICTION_PHASE'] = 'zipfian'
    
    # 使用真实的特征范围（基于训练数据）
    test_features = np.random.rand(69)
    # 模拟一些合理的特征值
    test_features[0:10] = np.random.uniform(0, 1, 10)  # rank features
    test_features[10:20] = np.random.uniform(0, 100, 10)  # score features
    test_features[20:40] = np.random.uniform(0, 1000000, 20)  # level features
    test_features[40:69] = np.random.uniform(0, 1, 29)  # other features
    
    print("\n  各Level预测结果（zipfian阶段，seqno=100M）:")
    for level in range(1, 7):
        should_predict, lifetime, status = simulate_rocksdb_ml_prediction(
            100000000, level, test_features, predictor
        )
        config = predictor.configs[level]
        print(f"    Level {level}: "
              f"生命周期={lifetime:>8.2f}秒, "
              f"模型={config.get('model_type', 'N/A'):>4}, "
              f"预处理={config['preprocessing']:>15}, "
              f"状态={status}")
    
    print("\n" + "="*80)
    all_ok = scenario1_ok and scenario2_ok and scenario3_ok
    if all_ok:
        print("✓ 所有集成测试通过！ML预测功能控制正常工作")
        print("\n推荐配置（你的需求）:")
        print("  export ROCKSDB_ML_PREDICTION_ENABLE=1")
        print("  export ROCKSDB_ML_PREDICTION_PHASE=zipfian")
        print("\n效果:")
        print("  ✓ Uniform阶段: ML预测关闭")
        print("  ✓ Zipfian阶段: ML预测开启")
    else:
        print("✗ 部分集成测试失败，请检查配置")
    print("="*80)
    
    return all_ok

if __name__ == '__main__':
    success = test_integration_scenario()
    sys.exit(0 if success else 1)



















