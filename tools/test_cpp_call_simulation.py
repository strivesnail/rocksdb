#!/usr/bin/env python3
"""
模拟C++调用Python预测器的完整场景
测试与RocksDB实际调用场景的一致性
"""

import sys
import os
import numpy as np
from pathlib import Path

# 添加路径（模拟C++代码中的路径设置）
sys.path.insert(0, '/home/usr/test_environment/rocksdb/tools')

def simulate_cpp_call():
    """模拟C++调用Python预测器的完整流程"""
    print("="*80)
    print("模拟C++调用Python预测器测试")
    print("="*80)
    
    # 步骤1: 导入模块（模拟C++的PyImport_ImportModule）
    print("\n步骤1: 导入Python模块（模拟PyImport_ImportModule）...")
    try:
        g_ml_predict_module = __import__('ml_predict_lifetime_by_level')
        print("   ✓ 模块导入成功")
    except Exception as e:
        print(f"   ✗ 模块导入失败: {e}")
        return False
    
    # 步骤2: 获取initialize函数（模拟C++的PyObject_GetAttrString）
    print("\n步骤2: 获取initialize函数（模拟PyObject_GetAttrString）...")
    try:
        init_func = getattr(g_ml_predict_module, 'initialize')
        if callable(init_func):
            print("   ✓ initialize函数可调用")
        else:
            print("   ✗ initialize函数不可调用")
            return False
    except AttributeError:
        print("   ✗ initialize函数不存在")
        return False
    
    # 步骤3: 调用initialize函数（模拟C++的PyObject_CallObject）
    print("\n步骤3: 调用initialize函数（模拟PyObject_CallObject）...")
    try:
        init_result = init_func()
        if init_result:
            print("   ✓ initialize()返回True，初始化成功")
        else:
            print("   ✗ initialize()返回False，初始化失败")
            return False
    except Exception as e:
        print(f"   ✗ initialize()调用失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 步骤4: 获取predict函数（模拟C++的PyObject_GetAttrString）
    print("\n步骤4: 获取predict_file_lifetime_by_level函数...")
    try:
        predict_func = getattr(g_ml_predict_module, 'predict_file_lifetime_by_level')
        if callable(predict_func):
            print("   ✓ predict_file_lifetime_by_level函数可调用")
        else:
            print("   ✗ predict_file_lifetime_by_level函数不可调用")
            return False
    except AttributeError:
        print("   ✗ predict_file_lifetime_by_level函数不存在")
        return False
    
    # 步骤5: 模拟C++调用预测函数（模拟PyObject_CallObject）
    print("\n步骤5: 模拟C++调用预测函数（模拟PyObject_CallObject）...")
    
    # 创建69个特征的数组（模拟C++传递的double数组）
    features_array = np.random.rand(69).astype(np.float64)
    # 模拟一些合理的特征值
    features_array[0:10] = np.random.uniform(0, 1, 10)
    features_array[10:20] = np.random.uniform(0, 100, 10)
    features_array[20:40] = np.random.uniform(0, 1000000, 20)
    features_array[40:69] = np.random.uniform(0, 1, 29)
    
    test_cases = [
        (1, 51840002),  # zipfian阶段，Level 1
        (2, 55000000),  # zipfian阶段，Level 2
        (3, 56000000),  # zipfian阶段，Level 3
        (4, 55000000),  # zipfian阶段，Level 4
        (5, 56000000),  # zipfian阶段，Level 5
        (6, 56000000),  # zipfian阶段，Level 6
    ]
    
    success_count = 0
    for level, seqno in test_cases:
        try:
            # 模拟C++调用: predict_file_lifetime_by_level(features, level)
            predicted_lifetime = predict_func(features_array, level)
            
            if predicted_lifetime > 0:
                # 模拟C++的MapLifetimeToHint逻辑
                if predicted_lifetime < 10.0:
                    hint = "WLTH_LEVEL0"
                elif predicted_lifetime < 50.0:
                    hint = "WLTH_LEVEL1"
                elif predicted_lifetime < 100.0:
                    hint = "WLTH_LEVEL2"
                elif predicted_lifetime < 150.0:
                    hint = "WLTH_LEVEL3"
                elif predicted_lifetime < 200.0:
                    hint = "WLTH_LEVEL4"
                elif predicted_lifetime < 400.0:
                    hint = "WLTH_LEVEL5"
                else:
                    hint = "WLTH_LEVEL6"
                
                print(f"   ✓ Level {level} (seqno={seqno:,}): "
                      f"生命周期={predicted_lifetime:.2f}秒 → {hint}")
                success_count += 1
            else:
                print(f"   ✗ Level {level}: 预测返回无效值 {predicted_lifetime}")
        except Exception as e:
            print(f"   ✗ Level {level}: 预测失败 - {e}")
            import traceback
            traceback.print_exc()
    
    if success_count == len(test_cases):
        print(f"\n   ✓ 所有 {len(test_cases)} 个测试用例都成功")
    else:
        print(f"\n   ✗ 只有 {success_count}/{len(test_cases)} 个测试用例成功")
        return False
    
    # 步骤6: 测试连续多次调用（模拟实际compaction场景）
    print("\n步骤6: 测试连续多次调用（模拟实际compaction场景）...")
    continuous_success = 0
    for i in range(20):
        try:
            features = np.random.rand(69).astype(np.float64)
            level = np.random.randint(1, 7)
            lifetime = predict_func(features, level)
            if lifetime > 0:
                continuous_success += 1
        except Exception as e:
            print(f"   第{i+1}次调用失败: {e}")
    
    print(f"   连续调用成功: {continuous_success}/20 次")
    if continuous_success == 20:
        print("   ✓ 连续调用测试通过")
    else:
        print(f"   ✗ 连续调用测试失败 ({continuous_success}/20)")
        return False
    
    print("\n" + "="*80)
    print("✓ 所有C++调用模拟测试通过！")
    print("="*80)
    print("\n结论:")
    print("  - Python模块可以正常导入")
    print("  - initialize()函数可以正常调用并返回True")
    print("  - predict_file_lifetime_by_level()函数可以正常预测")
    print("  - 所有Level的模型都能正常工作")
    print("  - 连续多次调用稳定")
    print("\nML预测功能已经可以正常使用！")
    print("="*80)
    
    return True

if __name__ == '__main__':
    success = simulate_cpp_call()
    sys.exit(0 if success else 1)


















