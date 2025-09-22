"""
模型评估工具
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import os

def evaluate_predictions(y_true, y_pred, title="模型评估结果"):
    """
    评估预测结果
    
    Args:
        y_true (array-like): 真实值
        y_pred (array-like): 预测值
        title (str): 图表标题
        
    Returns:
        dict: 评估指标
    """
    # 计算评估指标
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    metrics = {
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2
    }
    
    print(f"{title}:")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAE: {mae:.4f}")
    print(f"  R²: {r2:.4f}")
    
    return metrics

def plot_predictions(y_true, y_pred, title="预测值 vs 真实值"):
    """
    绘制预测值与真实值的对比图
    
    Args:
        y_true (array-like): 真实值
        y_pred (array-like): 预测值
        title (str): 图表标题
    """
    plt.figure(figsize=(10, 6))
    
    # 绘制散点图
    plt.scatter(y_true, y_pred, alpha=0.5)
    
    # 绘制完美预测线
    min_val = min(min(y_true), min(y_pred))
    max_val = max(max(y_true), max(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
    
    plt.xlabel('真实值')
    plt.ylabel('预测值')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    
    # 添加R2分数到图中
    r2 = r2_score(y_true, y_pred)
    plt.text(0.05, 0.95, f'R² = {r2:.4f}', transform=plt.gca().transAxes, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

def plot_residuals(y_true, y_pred, title="残差图"):
    """
    绘制残差图
    
    Args:
        y_true (array-like): 真实值
        y_pred (array-like): 预测值
        title (str): 图表标题
    """
    residuals = y_true - y_pred
    
    plt.figure(figsize=(10, 6))
    plt.scatter(y_pred, residuals, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('预测值')
    plt.ylabel('残差')
    plt.title(title)
    plt.grid(True, alpha=0.3)

def load_and_evaluate_model(model_path, test_data_path):
    """
    加载模型并对测试数据进行评估
    
    Args:
        model_path (str): 模型路径
        test_data_path (str): 测试数据路径
        
    Returns:
        tuple: (model, metrics)
    """
    # 加载模型
    model = joblib.load(model_path)
    
    # 加载测试数据
    test_data = pd.read_csv(test_data_path)
    
    # 分离特征和目标
    feature_columns = test_data.columns[:-1]  # 假设最后一列是目标变量
    target_column = test_data.columns[-1]
    
    X_test = test_data[feature_columns]
    y_test = test_data[target_column]
    
    # 预测
    y_pred = model.predict(X_test)
    
    # 评估
    metrics = evaluate_predictions(y_test, y_pred)
    
    # 绘图
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plot_predictions(y_test, y_pred)
    
    plt.subplot(1, 3, 2)
    plot_residuals(y_test, y_pred)
    
    plt.subplot(1, 3, 3)
    plt.hist(y_test - y_pred, bins=30, edgecolor='black', alpha=0.7)
    plt.xlabel('残差')
    plt.ylabel('频数')
    plt.title('残差分布')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return model, metrics