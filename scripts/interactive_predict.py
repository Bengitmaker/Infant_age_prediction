"""
交互式预测脚本
允许用户输入婴儿的信息并预测年龄
"""
import pandas as pd
import numpy as np
import joblib
import os
import sys
from datetime import datetime

# 添加项目根目录到sys.path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from configs.config_manager import config_manager

def get_user_input():
    """
    获取用户输入的婴儿信息
    
    Returns:
        pd.DataFrame: 包含用户输入信息的数据框
    """
    print("请输入婴儿的信息：")
    
    # 获取分类特征
    cat_id = input("cat_id (例如: 1001): ")
    cat1 = input("cat1 (例如: 2001): ")
    gender = input("gender (例如: 0 或 1): ")
    
    # 获取数值特征
    property_str = input("property (例如: '1001;21458;3002'): ")
    birthday_date_str = input("birthday_date (格式: YYYY-MM-DD, 例如: 2020-01-15): ")
    day_date_str = input("day_date (格式: YYYY-MM-DD, 例如: 2021-01-15): ")
    buy_mount = float(input("buy_mount (例如: 5): "))
    auction_id = int(input("auction_id (例如: 123456789): "))
    
    # 处理property特征
    if pd.isna(property_str) or property_str == "":
        property_count = 0
        has_special_property = 0
        sum_properties = 0
    else:
        properties = property_str.split(';')
        property_count = len(properties)
        has_special_property = 1 if '21458' in property_str else 0
        numbers = []
        for prop in properties:
            numbers.extend([int(num) for num in prop.split() if num.isdigit()])
        sum_properties = sum(numbers) if numbers else 0
    
    # 处理日期特征
    birthday_date = datetime.strptime(birthday_date_str, '%Y-%m-%d')
    day_date = datetime.strptime(day_date_str, '%Y-%m-%d')
    
    birthday_year = birthday_date.year
    birthday_month = birthday_date.month
    day_year = day_date.year
    day_month = day_date.month
    days_diff = (day_date - birthday_date).days
    
    # 计算对数特征
    buy_mount_log = np.log1p(buy_mount)
    
    # auction_id的后三位
    auction_id_last_digits = auction_id % 1000
    
    # 创建数据框
    data = pd.DataFrame({
        'cat_id': [cat_id],
        'cat1': [cat1],
        'gender': [gender],
        'property_count': [property_count],
        'has_special_property': [has_special_property],
        'sum_properties': [sum_properties],
        'birthday_year': [birthday_year],
        'birthday_month': [birthday_month],
        'day_year': [day_year],
        'day_month': [day_month],
        'days_diff': [days_diff],
        'buy_mount_log': [buy_mount_log],
        'auction_id_last_digits': [auction_id_last_digits]
    })
    
    return data

def predict_age_interactive():
    """
    交互式预测婴儿年龄
    """
    # 获取模型路径
    model_path = config_manager.get('output.model_path')
    model_path = os.path.join(project_root, model_path)
    
    # 检查模型文件是否存在
    if not os.path.exists(model_path):
        print(f"错误: 找不到模型文件 {model_path}")
        print("请先运行训练脚本: python scripts/train_model.py")
        return
    
    # 加载模型
    print(f"正在加载模型: {model_path}")
    model = joblib.load(model_path)
    
    # 获取用户输入
    user_data = get_user_input()
    
    # 进行预测
    print("正在进行预测...")
    prediction = model.predict(user_data)
    
    # 显示结果
    print(f"\n预测结果:")
    print(f"婴儿的年龄预测为: {prediction[0]:.2f} 天")
    print(f"婴儿的年龄预测为: {prediction[0]/30:.2f} 个月")
    print(f"婴儿的年龄预测为: {prediction[0]/365:.2f} 年")

if __name__ == "__main__":
    predict_age_interactive()