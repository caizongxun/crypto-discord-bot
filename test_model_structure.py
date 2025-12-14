#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
測試腳本：檢查模型結構和 checkpoint 內容
"""

import torch
from pathlib import Path
import json

def inspect_checkpoint(model_path):
    """檢查 checkpoint 的詳細結構"""
    print(f"\n{'='*80}")
    print(f"檢查: {model_path.name}")
    print(f"{'='*80}")
    
    try:
        checkpoint = torch.load(model_path, map_location='cpu')
        
        print(f"\n✓ 成功載入 checkpoint")
        print(f"\n📋 Checkpoint 包含的鍵:")
        
        # 分類顯示
        lstm_keys = {}
        regressor_keys = {}
        other_keys = {}
        
        for key in sorted(checkpoint.keys()):
            value = checkpoint[key]
            shape = value.shape if isinstance(value, torch.Tensor) else type(value)
            
            if 'lstm' in key:
                lstm_keys[key] = shape
            elif 'regressor' in key:
                regressor_keys[key] = shape
            else:
                other_keys[key] = shape
        
        # 顯示 LSTM 層
        if lstm_keys:
            print(f"\n🔹 LSTM 層 ({len(lstm_keys)} 個):")
            for key, shape in sorted(lstm_keys.items()):
                print(f"  {key}: {shape}")
        
        # 顯示 Regressor 層
        if regressor_keys:
            print(f"\n🔹 Regressor 層 ({len(regressor_keys)} 個):")
            for key, shape in sorted(regressor_keys.items()):
                print(f"  {key}: {shape}")
        
        # 顯示其他
        if other_keys:
            print(f"\n🔹 其他 ({len(other_keys)} 個):")
            for key, shape in sorted(other_keys.items()):
                print(f"  {key}: {shape}")
        
        # 統計資訊
        print(f"\n📊 統計:")
        print(f"  LSTM 層: {len(lstm_keys)} 個鍵")
        print(f"  Regressor 層: {len(regressor_keys)} 個鍵")
        print(f"  總計: {len(checkpoint)} 個鍵")
        
        # 分析 LSTM 結構
        print(f"\n🔬 LSTM 結構分析:")
        bidirectional = any('reverse' in k for k in lstm_keys.keys())
        print(f"  Bidirectional: {bidirectional}")
        
        # 從 weight_ih_l0 推斷輸入大小
        if 'lstm.weight_ih_l0' in lstm_keys:
            weight_ih_shape = lstm_keys['lstm.weight_ih_l0']
            # weight_ih 的形狀是 (gates * hidden_size, input_size)
            # gates = 4 (input, forget, cell, output)
            gates = 4
            hidden_size = weight_ih_shape[0] // gates
            input_size = weight_ih_shape[1]
            print(f"  Hidden size: {hidden_size}")
            print(f"  Input size: {input_size}")
            print(f"  Weight_ih shape: {weight_ih_shape}")
        
        # 分析 Regressor 結構
        print(f"\n🔬 Regressor 結構分析:")
        regressor_indices = []
        for key in regressor_keys.keys():
            # 提取層索引: regressor.0.weight -> 0
            parts = key.split('.')
            if len(parts) >= 2:
                try:
                    idx = int(parts[1])
                    if idx not in regressor_indices:
                        regressor_indices.append(idx)
                except:
                    pass
        
        regressor_indices.sort()
        print(f"  有參數的層索引: {regressor_indices}")
        
        # 推斷層結構
        if regressor_indices:
            print(f"  層結構 (推測):")
            for i, idx in enumerate(regressor_indices):
                if f'regressor.{idx}.weight' in regressor_keys:
                    weight_shape = regressor_keys[f'regressor.{idx}.weight']
                    print(f"    層 {idx}: Linear{weight_shape}")
        
        return checkpoint, lstm_keys, regressor_keys
    
    except Exception as e:
        print(f"✗ 錯誤: {e}")
        return None, None, None


def test_model_loading(checkpoint_path, checkpoint_data):
    """測試模型載入"""
    print(f"\n{'='*80}")
    print(f"測試模型載入")
    print(f"{'='*80}\n")
    
    # 從 checkpoint 推斷模型參數
    lstm_keys = {k: v for k, v in checkpoint_data.items() if 'lstm' in k}
    regressor_keys = {k: v for k, v in checkpoint_data.items() if 'regressor' in k}
    
    # 推斷 LSTM 參數
    if 'lstm.weight_ih_l0' in lstm_keys:
        weight_ih_shape = lstm_keys['lstm.weight_ih_l0']
        hidden_size = weight_ih_shape[0] // 4
        input_size = weight_ih_shape[1]
    else:
        print("✗ 找不到 lstm.weight_ih_l0")
        return
    
    # 推斷 Regressor 參數
    regressor_indices = set()
    for key in regressor_keys.keys():
        parts = key.split('.')
        if len(parts) >= 2:
            try:
                idx = int(parts[1])
                regressor_indices.add(idx)
            except:
                pass
    
    print(f"推斷的模型參數:")
    print(f"  Input size: {input_size}")
    print(f"  Hidden size: {hidden_size}")
    print(f"  Regressor 層索引: {sorted(regressor_indices)}")
    
    # 嘗試不同的模型結構
    print(f"\n嘗試模型結構...\n")
    
    # 方案 1: 簡單 Sequential
    print("方案 1: 簡單 Sequential (層索引 0-5)")
    try:
        model1 = torch.nn.Sequential(
            torch.nn.Linear(hidden_size * 2, 64),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(64, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 1)
        )
        # 檢查 regressor state dict
        regressor_state = {k: v for k, v in checkpoint_data.items() if 'regressor' in k}
        missing = set(regressor_state.keys()) - set(f'regressor.{k}' for k in model1.state_dict().keys())
        unexpected = set(f'regressor.{k}' for k in model1.state_dict().keys()) - set(regressor_state.keys())
        
        if missing or unexpected:
            print(f"  ✗ 不匹配")
            if missing:
                print(f"    缺少的鍵: {missing}")
            if unexpected:
                print(f"    多餘的鍵: {unexpected}")
        else:
            print(f"  ✓ 完全匹配!")
    except Exception as e:
        print(f"  ✗ 錯誤: {e}")
    
    # 方案 2: 用 ModuleList
    print(f"\n方案 2: ModuleList (只有索引 {sorted(regressor_indices)})")
    try:
        regressor_modules = torch.nn.ModuleList()
        max_idx = max(regressor_indices) if regressor_indices else 0
        
        # 按索引填充
        for i in range(max_idx + 1):
            if i == 0:
                regressor_modules.append(torch.nn.Linear(hidden_size * 2, 64))
            elif i == 1:
                regressor_modules.append(torch.nn.ReLU())
            elif i == 2:
                regressor_modules.append(torch.nn.Dropout(0.2))
            elif i == 3:
                regressor_modules.append(torch.nn.Linear(64, 32))
            elif i == 4:
                regressor_modules.append(torch.nn.ReLU())
            elif i == 5:
                regressor_modules.append(torch.nn.Linear(32, 1))
            else:
                regressor_modules.append(torch.nn.Identity())
        
        # 檢查
        print(f"  ✓ 建立成功 (6 層)")
    except Exception as e:
        print(f"  ✗ 錯誤: {e}")


def main():
    """主函數"""
    models_dir = Path('models')
    
    if not models_dir.exists():
        print(f"✗ 模型目錄不存在: {models_dir}")
        return
    
    model_files = sorted(models_dir.glob('*_model_v8.pth'))
    print(f"\n找到 {len(model_files)} 個模型文件\n")
    
    if not model_files:
        print("✗ 沒有找到模型文件")
        return
    
    # 檢查前 3 個模型
    for model_file in model_files[:3]:
        checkpoint, lstm_keys, regressor_keys = inspect_checkpoint(model_file)
        if checkpoint:
            test_model_loading(model_file, checkpoint)
    
    # 總結
    print(f"\n{'='*80}")
    print("總結")
    print(f"{'='*80}\n")
    
    print("根據上面的檢查結果:")
    print("\n請檢查 Regressor 層索引是否一致")
    print("如果層索引不是 0, 1, 2, 3, 4, 5 的連續序列")
    print("需要相應調整模型結構\n")


if __name__ == '__main__':
    main()
