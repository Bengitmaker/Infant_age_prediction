"""
预测脚本
"""
import pandas as pd
import numpy as np
import argparse
import joblib
import os
import sys

# 添加项目根目录到sys.path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from configs.config_manager import config_manager

def predict_age(data_path=None, model_path=None, output_path=None):
    """
    使用训练好的模型进行预测
    
    Args:
        data_path (str): 数据路径
        model_path (str): 模型路径
        output_path (str): 预测结果保存路径
    """
    # 获取配置
    if data_path is None:
        data_path = config_manager.get('data.processed_data_path')
        # 转换为绝对路径
        data_path = os.path.join(project_root, data_path)
    if model_path is None:
        model_path = config_manager.get('output.model_path')
        # 转换为绝对路径
        model_path = os.path.join(project_root, model_path)
    if output_path is None:
        output_path = config_manager.get('output.predictions_path')
        # 转换为绝对路径
        output_path = os.path.join(project_root, output_path)
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 读取数据
    print(f"正在读取数据: {data_path}")
    data = pd.read_csv(data_path)
    print(f"数据形状: {data.shape}")
    
    # 获取特征列
    categorical_features = config_manager.get('features.categorical')
    numerical_features = config_manager.get('features.numerical')
    feature_columns = categorical_features + numerical_features
    
    # 数据清洗（删除缺失值）
    data_clean = data.dropna()
    print(f"清洗后数据形状: {data_clean.shape}")
    
    X = data_clean[feature_columns]
    
    # 加载模型
    print(f"正在加载模型: {model_path}")
    model = joblib.load(model_path)
    
    # 预测
    print("正在进行预测...")
    predictions = model.predict(X)
    
    # 保存预测结果
    results = pd.DataFrame({
        'prediction': predictions
    })
    
    print(f"正在保存预测结果到: {output_path}")
    results.to_csv(output_path, index=False)
    
    print("预测完成!")
    print(f"预测值范围: {predictions.min():.2f} - {predictions.max():.2f}")
    
    return predictions

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='使用模型进行年龄预测')
    parser.add_argument('--data', type=str, help='数据路径')
    parser.add_argument('--model', type=str, help='模型路径')
    parser.add_argument('--output', type=str, help='预测结果保存路径')
    
    args = parser.parse_args()
    
    predict_age(args.data, args.model, args.output)