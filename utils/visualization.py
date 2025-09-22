"""
数据可视化工具
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_age_distribution(data, column='age', title='年龄分布'):
    """
    绘制年龄分布图
    
    Args:
        data (pd.DataFrame): 数据
        column (str): 年龄列名
        title (str): 图表标题
    """
    plt.figure(figsize=(10, 6))
    plt.hist(data[column], bins=50, edgecolor='black', alpha=0.7)
    plt.xlabel('年龄（月）')
    plt.ylabel('频数')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.show()

def plot_feature_importance(feature_names, importances, top_n=10):
    """
    绘制特征重要性图
    
    Args:
        feature_names (list): 特征名称列表
        importances (array): 特征重要性数组
        top_n (int): 显示前N个重要特征
    """
    # 创建特征重要性DataFrame
    feature_importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False)
    
    # 选择前N个特征
    top_features = feature_importance_df.head(top_n)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(data=top_features, x='importance', y='feature')
    plt.title(f'前{top_n}个重要特征')
    plt.xlabel('重要性')
    plt.tight_layout()
    plt.show()
    
    return top_features

def plot_correlation_matrix(data, figsize=(12, 10)):
    """
    绘制相关性矩阵热力图
    
    Args:
        data (pd.DataFrame): 数据
        figsize (tuple): 图表大小
    """
    # 计算相关性矩阵
    corr_matrix = data.corr()
    
    plt.figure(figsize=figsize)
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                fmt='.2f', square=True, linewidths=0.5)
    plt.title('特征相关性矩阵')
    plt.tight_layout()
    plt.show()
    
    return corr_matrix

def plot_categorical_feature_distribution(data, feature, target='age'):
    """
    绘制分类特征的分布图
    
    Args:
        data (pd.DataFrame): 数据
        feature (str): 分类特征名称
        target (str): 目标变量名称
    """
    plt.figure(figsize=(12, 6))
    
    # 子图1: 分类特征的计数
    plt.subplot(1, 2, 1)
    data[feature].value_counts().plot(kind='bar')
    plt.title(f'{feature} 分布')
    plt.xticks(rotation=45)
    
    # 子图2: 分类特征与目标变量的关系
    plt.subplot(1, 2, 2)
    data.groupby(feature)[target].mean().plot(kind='bar')
    plt.title(f'{feature} vs 平均{target}')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.show()