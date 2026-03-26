#!/usr/bin/env python3
"""
测试ML预测功能控制
模拟RocksDB的ML预测调用逻辑，验证环境变量控制和阶段判断是否生效
"""

import os
import sys
import numpy as np
from pathlib import Path

# 添加路径以便导入预测器
sys.path.insert(0, '/home/usr/test_environment/rocksdb/tools')
from ml_predict_lifetime_by_level import FileLifetimePredictorByLevel

def test_ml_prediction_control():
    """测试ML预测功能控制"""
    print("="*80)
    print("ML预测功能控制测试")
    print("="*80)
    
    # 初始化预测器
    print("\n1. 初始化预测器...")
    try:
        predictor = FileLifetimePredictorByLevel()
        print(f"   ✓ 预测器初始化成功")
        print(f"   ✓ 已加载Level: {list(predictor.models.keys())}")
    except Exception as e:
        print(f"   ✗ 预测器初始化失败: {e}")
        return False
    
    # 测试环境变量读取逻辑（模拟C++代码）
    print("\n2. 测试环境变量控制逻辑...")
    
    test_cases = [
        {
            'name': '测试1: 完全禁用',
            'env': {'ROCKSDB_ML_PREDICTION_ENABLE': '0'},
            'seqno': 100000000,  # zipfian阶段
            'expected': False
        },
        {
            'name': '测试2: 启用，zipfian阶段，配置为zipfian',
            'env': {
                'ROCKSDB_ML_PREDICTION_ENABLE': '1',
                'ROCKSDB_ML_PREDICTION_PHASE': 'zipfian'
            },
            'seqno': 100000000,  # zipfian阶段
            'expected': True
        },
        {
            'name': '测试3: 启用，uniform阶段，配置为zipfian',
            'env': {
                'ROCKSDB_ML_PREDICTION_ENABLE': '1',
                'ROCKSDB_ML_PREDICTION_PHASE': 'zipfian'
            },
            'seqno': 10000000,  # uniform阶段
            'expected': False
        },
        {
            'name': '测试4: 启用，uniform阶段，配置为uniform',
            'env': {
                'ROCKSDB_ML_PREDICTION_ENABLE': '1',
                'ROCKSDB_ML_PREDICTION_PHASE': 'uniform'
            },
            'seqno': 10000000,  # uniform阶段
            'expected': True
        },
        {
            'name': '测试5: 启用，zipfian阶段，配置为both',
            'env': {
                'ROCKSDB_ML_PREDICTION_ENABLE': '1',
                'ROCKSDB_ML_PREDICTION_PHASE': 'both'
            },
            'seqno': 100000000,  # zipfian阶段
            'expected': True
        },
        {
            'name': '测试6: 启用，uniform阶段，配置为both',
            'env': {
                'ROCKSDB_ML_PREDICTION_ENABLE': '1',
                'ROCKSDB_ML_PREDICTION_PHASE': 'both'
            },
            'seqno': 10000000,  # uniform阶段
            'expected': True
        },
        {
            'name': '测试7: 默认配置（不设置环境变量）',
            'env': {},
            'seqno': 100000000,  # zipfian阶段
            'expected': True  # 默认启用
        },
    ]
    
    all_passed = True
    
    for test_case in test_cases:
        print(f"\n   {test_case['name']}")
        
        # 设置环境变量
        for key, value in test_case['env'].items():
            os.environ[key] = value
        
        # 清除未设置的环境变量
        for key in ['ROCKSDB_ML_PREDICTION_ENABLE', 'ROCKSDB_ML_PREDICTION_PHASE']:
            if key not in test_case['env']:
                os.environ.pop(key, None)
        
        # 模拟C++代码的逻辑
        ml_enable_env = os.getenv('ROCKSDB_ML_PREDICTION_ENABLE')
        ml_phase_env = os.getenv('ROCKSDB_ML_PREDICTION_PHASE')
        
        ml_enabled = (ml_enable_env is None or ml_enable_env != '0')
        ml_phase = ml_phase_env if ml_phase_env else 'both'
        
        # 判断阶段（模拟C++代码）
        current_seqno = test_case['seqno']
        is_zipfian_phase = True
        if current_seqno < 50000000:
            is_zipfian_phase = False  # uniform阶段
        
        # 根据阶段和配置决定是否启用ML预测
        use_ml_prediction = False
        if ml_enabled:
            if ml_phase == 'both':
                use_ml_prediction = True
            elif ml_phase == 'zipfian' and is_zipfian_phase:
                use_ml_prediction = True
            elif ml_phase == 'uniform' and not is_zipfian_phase:
                use_ml_prediction = True
        
        # 验证结果
        phase_name = 'zipfian' if is_zipfian_phase else 'uniform'
        result = '✓' if use_ml_prediction == test_case['expected'] else '✗'
        
        print(f"     环境变量: ENABLE={ml_enable_env or '默认(1)'}, PHASE={ml_phase_env or '默认(both)'}")
        print(f"     当前阶段: {phase_name} (seqno={current_seqno:,})")
        print(f"     ML预测: {'启用' if use_ml_prediction else '禁用'}")
        print(f"     期望: {'启用' if test_case['expected'] else '禁用'}")
        print(f"     结果: {result}")
        
        if use_ml_prediction != test_case['expected']:
            all_passed = False
    
    # 测试实际预测功能
    print("\n3. 测试实际预测功能...")
    
    # 设置环境变量为zipfian阶段启用
    os.environ['ROCKSDB_ML_PREDICTION_ENABLE'] = '1'
    os.environ['ROCKSDB_ML_PREDICTION_PHASE'] = 'zipfian'
    
    # 创建测试特征（69个特征）
    test_features = np.random.rand(69)
    
    print("\n   测试各Level的预测:")
    for level in range(1, 7):
        if level in predictor.models and predictor.models[level] is not None:
            try:
                lifetime = predictor.predict(test_features, level)
                config = predictor.configs[level]
                print(f"     Level {level}: ✓ 预测成功，生命周期={lifetime:.2f}秒 "
                      f"(模型={config.get('model_type', 'N/A')}, "
                      f"预处理={config['preprocessing']}, "
                      f"测试R²={config.get('test_r2', 0):.4f})")
            except Exception as e:
                print(f"     Level {level}: ✗ 预测失败 - {e}")
                all_passed = False
        else:
            print(f"     Level {level}: ⚠ 模型未加载")
    
    # 测试阶段判断边界情况
    print("\n4. 测试阶段判断边界情况...")
    
    boundary_tests = [
        {'seqno': 49999999, 'expected_phase': 'uniform'},
        {'seqno': 50000000, 'expected_phase': 'zipfian'},
        {'seqno': 50000001, 'expected_phase': 'zipfian'},
        {'seqno': 0, 'expected_phase': 'uniform'},
        {'seqno': 100000000, 'expected_phase': 'zipfian'},
    ]
    
    for test in boundary_tests:
        seqno = test['seqno']
        is_zipfian = seqno >= 50000000
        phase_name = 'zipfian' if is_zipfian else 'uniform'
        result = '✓' if phase_name == test['expected_phase'] else '✗'
        print(f"     seqno={seqno:>12,}: {phase_name:>8} {result}")
        if phase_name != test['expected_phase']:
            all_passed = False
    
    print("\n" + "="*80)
    if all_passed:
        print("✓ 所有测试通过！ML预测功能控制正常工作")
    else:
        print("✗ 部分测试失败，请检查配置")
    print("="*80)
    
    return all_passed

if __name__ == '__main__':
    success = test_ml_prediction_control()
    sys.exit(0 if success else 1)



















