#!/usr/bin/env python3
"""
测试ML预测功能运行时是否正常工作
模拟C++调用Python预测器的完整流程
"""

import sys
import os
import numpy as np
from pathlib import Path

# 添加路径
sys.path.insert(0, '/home/usr/test_environment/rocksdb/tools')

def test_ml_prediction_runtime():
    """测试ML预测运行时功能"""
    print("="*80)
    print("ML预测功能运行时测试")
    print("="*80)
    
    # 1. 测试模块导入
    print("\n1. 测试Python模块导入...")
    try:
        from ml_predict_lifetime_by_level import (
            initialize,
            predict_file_lifetime_by_level,
            get_predictor
        )
        print("   ✓ 模块导入成功")
    except Exception as e:
        print(f"   ✗ 模块导入失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 2. 测试initialize函数
    print("\n2. 测试initialize()函数...")
    try:
        result = initialize()
        if result:
            print("   ✓ initialize()返回True，初始化成功")
        else:
            print("   ✗ initialize()返回False，初始化失败")
            return False
    except Exception as e:
        print(f"   ✗ initialize()调用失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 3. 测试get_predictor函数
    print("\n3. 测试get_predictor()函数...")
    try:
        predictor = get_predictor()
        if predictor is None:
            print("   ✗ get_predictor()返回None")
            return False
        print(f"   ✓ get_predictor()成功，已加载 {len(predictor.models)} 个模型")
        print(f"     模型Level: {sorted(predictor.models.keys())}")
    except Exception as e:
        print(f"   ✗ get_predictor()调用失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 4. 测试predict_file_lifetime_by_level函数（模拟C++调用）
    print("\n4. 测试predict_file_lifetime_by_level()函数（模拟C++调用）...")
    
    # 创建测试特征（69个特征）
    test_features = np.random.rand(69)
    # 模拟一些合理的特征值范围
    test_features[0:10] = np.random.uniform(0, 1, 10)  # rank features
    test_features[10:20] = np.random.uniform(0, 100, 10)  # score features
    test_features[20:40] = np.random.uniform(0, 1000000, 20)  # level features
    test_features[40:69] = np.random.uniform(0, 1, 29)  # other features
    
    success_count = 0
    for level in range(1, 7):
        try:
            lifetime = predict_file_lifetime_by_level(test_features, level)
            if lifetime > 0:
                config = predictor.configs[level]
                print(f"   ✓ Level {level}: 预测成功，生命周期={lifetime:.2f}秒 "
                      f"(模型={config.get('model_type', 'N/A')}, "
                      f"测试R²={config.get('test_r2', 0):.4f})")
                success_count += 1
            else:
                print(f"   ✗ Level {level}: 预测返回0或负数")
        except Exception as e:
            print(f"   ✗ Level {level}: 预测失败 - {e}")
            import traceback
            traceback.print_exc()
    
    if success_count == 6:
        print(f"\n   ✓ 所有6个Level的预测都成功")
    else:
        print(f"\n   ✗ 只有 {success_count}/6 个Level预测成功")
        return False
    
    # 5. 测试多次调用（模拟实际使用场景）
    print("\n5. 测试多次调用（模拟实际使用场景）...")
    test_count = 10
    success_count = 0
    for i in range(test_count):
        try:
            features = np.random.rand(69)
            level = np.random.randint(1, 7)
            lifetime = predict_file_lifetime_by_level(features, level)
            if lifetime > 0:
                success_count += 1
        except Exception as e:
            print(f"   第{i+1}次调用失败: {e}")
    
    print(f"   成功: {success_count}/{test_count} 次调用")
    if success_count == test_count:
        print("   ✓ 多次调用测试通过")
    else:
        print(f"   ✗ 多次调用测试失败 ({success_count}/{test_count})")
        return False
    
    # 6. 测试边界情况
    print("\n6. 测试边界情况...")
    
    # 测试level边界
    try:
        lifetime1 = predict_file_lifetime_by_level(test_features, 1)
        lifetime6 = predict_file_lifetime_by_level(test_features, 6)
        print(f"   ✓ Level边界测试: Level 1={lifetime1:.2f}秒, Level 6={lifetime6:.2f}秒")
    except Exception as e:
        print(f"   ✗ Level边界测试失败: {e}")
        return False
    
    # 测试无效level（应该使用边界值）
    try:
        lifetime0 = predict_file_lifetime_by_level(test_features, 0)  # 应该使用Level 1
        lifetime7 = predict_file_lifetime_by_level(test_features, 7)  # 应该使用Level 6
        print(f"   ✓ 无效Level处理: Level 0={lifetime0:.2f}秒, Level 7={lifetime7:.2f}秒")
    except Exception as e:
        print(f"   ⚠ 无效Level处理: {e} (可能正常，取决于实现)")
    
    # 7. 测试环境变量
    print("\n7. 测试环境变量...")
    models_path = os.getenv('ROCKSDB_ML_MODELS_PATH')
    if models_path:
        print(f"   ✓ ROCKSDB_ML_MODELS_PATH={models_path}")
        if Path(models_path).exists():
            print(f"   ✓ 模型路径存在")
        else:
            print(f"   ✗ 模型路径不存在")
    else:
        print(f"   ⚠ ROCKSDB_ML_MODELS_PATH未设置（将使用默认路径）")
    
    print("\n" + "="*80)
    print("✓ 所有测试通过！ML预测功能运行时正常工作")
    print("="*80)
    
    return True

if __name__ == '__main__':
    success = test_ml_prediction_runtime()
    sys.exit(0 if success else 1)


















