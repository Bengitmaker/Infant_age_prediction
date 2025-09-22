"""
数据处理脚本
"""
import pandas as pd
import numpy as np
import re
import argparse
import sys
import os

# 添加项目根目录到sys.path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from datetime import datetime
from configs.config_manager import config_manager

def extract_property_features(property_str):
    """
    从property字段中提取特征
    
    Args:
        property_str (str): 属性字符串
        
    Returns:
        pd.Series: 包含属性特征的Series
    """
    if pd.isna(property_str):
        return pd.Series([0, 0, 0])
    
    # 计算属性数量
    property_count = len(property_str.split(';'))
    
    # 检查是否包含特定关键词
    has_special_property = 1 if '21458' in property_str else 0  # 假设21458是重要属性
    
    # 计算数字属性值的总和
    numbers = re.findall(r'\d+', property_str)
    sum_properties = sum(int(num) for num in numbers) if numbers else 0
    
    return pd.Series([property_count, has_special_property, sum_properties])

def extract_date_features(data):
    """
    从日期字段中提取特征
    
    Args:
        data (pd.DataFrame): 原始数据
        
    Returns:
        pd.DataFrame: 添加了日期特征的数据
    """
    # 将日期字符串转换为datetime对象
    # 不再使用birthday_date，因为我们无法得知用户生日
    data['day_date'] = pd.to_datetime(data['day_date'])
    
    # 提取年、月、日特征 (仅从day_date)
    data['day_year'] = data['day_date'].dt.year
    data['day_month'] = data['day_date'].dt.month
    
    # 不再计算日期差异，因为我们无法得知用户生日
    # data['days_diff'] = (data['day_date'] - data['birthday_date']).dt.days
    
    return data

def process_data(input_path=None, output_path=None):
    """
    处理婴儿年龄数据，提取特征
    
    Args:
        input_path (str): 输入数据路径
        output_path (str): 输出数据路径
    """
    # 获取配置
    if input_path is None:
        input_path = config_manager.get('data.raw_data_path')
        # 转换为绝对路径
        input_path = os.path.join(project_root, input_path)
    if output_path is None:
        output_path = config_manager.get('data.processed_data_path')
        # 转换为绝对路径
        output_path = os.path.join(project_root, output_path)
    
    # 读取数据
    print(f"正在读取数据: {input_path}")
    data = pd.read_csv(input_path)
    print(f"原始数据形状: {data.shape}")
    
    # 基础分类特征
    cat_cols = config_manager.get('features.categorical')
    target = 'age'
    
    # 从property字段提取特征
    print("正在提取属性特征...")
    property_features = data['property'].apply(extract_property_features)
    property_features.columns = ['property_count', 'has_special_property', 'sum_properties']
    
    # 从日期字段提取特征
    print("正在提取日期特征...")
    data = extract_date_features(data)
    
    # 创建新的数值特征
    print("正在创建数值特征...")
    data['buy_mount_log'] = np.log1p(data['buy_mount'])  # 对buy_mount取对数
    data['auction_id_last_digits'] = data['auction_id'] % 1000  # auction_id的后三位
    
    # 合并所有特征
    # 移除与birthday相关的特征
    feature_columns = cat_cols + ['property_count', 'has_special_property', 'sum_properties', 
                                 'day_year', 'day_month', 
                                 'buy_mount_log', 'auction_id_last_digits']
    
    # 构建最终数据集
    extended_data = pd.concat([data, property_features], axis=1)
    final_data = pd.concat([extended_data[feature_columns], extended_data[target]], axis=1)
    
    # 保存处理后的数据
    print(f"正在保存处理后的数据到: {output_path}")
    final_data.to_csv(output_path, index=False)
    print(f"处理完成，最终数据形状: {final_data.shape}")
    
    return final_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='处理婴儿年龄预测数据')
    parser.add_argument('--input', type=str, help='输入数据路径')
    parser.add_argument('--output', type=str, help='输出数据路径')
    
    args = parser.parse_args()
    
    process_data(args.input, args.output)