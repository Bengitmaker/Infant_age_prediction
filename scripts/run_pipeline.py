"""
主运行脚本
"""
import argparse
import os
import sys

# 添加项目根目录到sys.path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from scripts.process_data import process_data
from scripts.train_model import train_model
from scripts.predict import predict_age
from configs.config_manager import config_manager

def main():
    """
    运行完整的婴儿年龄预测流水线
    """
    print("开始运行婴儿年龄预测流水线...")
    
    # 获取配置
    raw_data_path = config_manager.get('data.raw_data_path')
    raw_data_path = os.path.join(project_root, raw_data_path)
    
    processed_data_path = config_manager.get('data.processed_data_path')
    processed_data_path = os.path.join(project_root, processed_data_path)
    
    model_path = config_manager.get('output.model_path')
    model_path = os.path.join(project_root, model_path)
    
    metrics_path = config_manager.get('output.metrics_path')
    metrics_path = os.path.join(project_root, metrics_path)
    
    predictions_path = config_manager.get('output.predictions_path')
    predictions_path = os.path.join(project_root, predictions_path)
    
    # 步骤1: 数据处理
    print("\n=== 步骤1: 数据处理 ===")
    process_data(raw_data_path, processed_data_path)
    
    # 步骤2: 模型训练
    print("\n=== 步骤2: 模型训练 ===")
    model, metrics = train_model(processed_data_path, model_path, metrics_path)
    
    # 步骤3: 预测
    print("\n=== 步骤3: 预测 ===")
    predictions = predict_age(processed_data_path, model_path, predictions_path)
    
    print("\n=== 流水线执行完成 ===")
    print(f"模型已保存到: {model_path}")
    print(f"指标已保存到: {metrics_path}")
    print(f"预测结果已保存到: {predictions_path}")

if __name__ == "__main__":
    main()