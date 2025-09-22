"""
模型训练脚本
"""
import pandas as pd
import numpy as np
import argparse
import json
import os
import sys

# 添加项目根目录到sys.path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingRegressor
import joblib
from configs.config_manager import config_manager

def build_model():
    """
    构建模型
    
    Returns:
        Pipeline: 构建好的模型管道
    """
    # 获取特征列
    categorical_features = config_manager.get('features.categorical')
    numerical_features = config_manager.get('features.numerical')
    
    # 列变换：One-Hot编码用于分类特征，数值特征保持不变
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
            ('num', 'passthrough', numerical_features)
        ])
    
    # 获取模型参数
    model_params = config_manager.get('model.params')
    
    # 创建模型
    if config_manager.get('model.type') == 'GradientBoostingRegressor':
        regressor = GradientBoostingRegressor(**model_params)
    else:
        raise ValueError(f"不支持的模型类型: {config_manager.get('model.type')}")
    
    # 创建管道
    model = Pipeline(steps=[('prep', preprocessor), ('reg', regressor)])
    return model

def train_model(data_path=None, model_path=None, metrics_path=None):
    """
    训练模型
    
    Args:
        data_path (str): 数据路径
        model_path (str): 模型保存路径
        metrics_path (str): 指标保存路径
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
    if metrics_path is None:
        metrics_path = config_manager.get('output.metrics_path')
        # 转换为绝对路径
        metrics_path = os.path.join(project_root, metrics_path)
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
    
    # 读取数据
    print(f"正在读取数据: {data_path}")
    data = pd.read_csv(data_path)
    print(f"数据形状: {data.shape}")
    
    # 准备数据
    categorical_features = config_manager.get('features.categorical')
    numerical_features = config_manager.get('features.numerical')
    feature_columns = categorical_features + numerical_features
    target = 'age'
    
    # 数据清洗（删除缺失值）
    data_clean = data.dropna()
    print(f"清洗后数据形状: {data_clean.shape}")
    
    X = data_clean[feature_columns]
    y = data_clean[target]
    
    # 数据划分
    test_size = config_manager.get('data.test_size')
    random_state = config_manager.get('data.random_state')
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state)
    
    print(f"训练集大小: {X_train.shape}")
    print(f"测试集大小: {X_test.shape}")
    
    # 构建和训练模型
    print("正在构建模型...")
    model = build_model()
    
    print("正在训练模型...")
    model.fit(X_train, y_train)
    
    # 预测
    print("正在预测...")
    predictions = model.predict(X_test)
    
    # 评估
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)
    
    metrics = {
        'rmse': float(rmse),
        'r2_score': float(r2),
        'train_size': len(X_train),
        'test_size': len(X_test)
    }
    
    print("模型性能:")
    print(f"  RMSE = {rmse:.2f}")
    print(f"  R² Score = {r2:.4f}")
    
    # 保存模型
    print(f"正在保存模型到: {model_path}")
    joblib.dump(model, model_path)
    
    # 保存指标
    print(f"正在保存指标到: {metrics_path}")
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    return model, metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='训练婴儿年龄预测模型')
    parser.add_argument('--data', type=str, help='数据路径')
    parser.add_argument('--model', type=str, help='模型保存路径')
    parser.add_argument('--metrics', type=str, help='指标保存路径')
    
    args = parser.parse_args()
    
    train_model(args.data, args.model, args.metrics)