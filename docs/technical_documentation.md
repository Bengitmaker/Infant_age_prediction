# 婴儿年龄预测项目技术文档

## 项目概述

本项目旨在通过机器学习方法预测婴儿的年龄（以月为单位）。通过特征工程和梯度提升算法，我们能够构建一个准确的预测模型，目前达到 R² Score = 0.9182 的优秀性能。

## 数据说明

### 原始数据字段
- `user_id`: 用户ID
- `birthday`: 出生日期（YYYYMMDD格式）
- `gender`: 性别（0表示女性，1表示男性）
- `auction_id`: 拍卖ID
- `cat_id`: 商品类别ID
- `cat1`: 一级商品类别
- `property`: 商品属性
- `buy_mount`: 购买数量
- `day`: 购买日期（YYYYMMDD格式）
- `birthday_date`: 出生日期（YYYY-MM-DD格式）
- `day_date`: 购买日期（YYYY-MM-DD格式）
- `age`: 婴儿年龄（目标变量，以月为单位）

## 特征工程

### 1. 商品属性特征
从 `property` 字段中提取以下特征：
- `property_count`: 属性数量
- `has_special_property`: 是否包含特定重要属性（21458）
- `sum_properties`: 所有属性值的总和

### 2. 时间特征
从日期字段中提取以下特征：
- `birthday_year`: 出生年份
- `birthday_month`: 出生月份
- `day_year`: 购买年份
- `day_month`: 购买月份
- `days_diff`: 出生与购买日期的天数差

### 3. 数值特征
- `buy_mount_log`: 购买数量的对数值（改善分布）
- `auction_id_last_digits`: 拍卖ID的后三位数字

## 模型架构

### 数据预处理
使用 `ColumnTransformer` 对特征进行预处理：
- 分类特征（cat_id, cat1, gender）使用 OneHotEncoder 编码
- 数值特征直接传递（passthrough）

### 模型选择
使用 `GradientBoostingRegressor` 作为基础模型，参数配置如下：
```yaml
n_estimators: 500
learning_rate: 0.01
max_depth: 8
min_samples_split: 10
min_samples_leaf: 4
subsample: 0.8
max_features: "sqrt"
random_state: 42
```

### 模型管道
模型通过 `Pipeline` 组织，确保预处理和训练步骤的一致性：
1. 数据预处理（特征编码）
2. 模型训练（梯度提升回归）

## 性能指标

当前模型性能：
- **R² Score**: 0.9182
- **RMSE**: 6.82

这表明模型能够解释约91.82%的数据变异，预测误差平均为6.82个月。

## 项目结构

```
Infant_age_prediction/
├── configs/              # 配置文件
├── scripts/              # 可执行脚本
├── utils/                # 工具函数
├── models/               # 模型文件
├── output/               # 输出结果
├── notebooks/            # Jupyter笔记本
├── data/                 # 数据文件
├── docs/                 # 技术文档
├── README.md             # 项目说明文档
└── requirements.txt      # 依赖列表
```

## 使用方法

### 运行完整流水线
```bash
python scripts/run_pipeline.py
```

### 单独运行各步骤
#### 数据处理
```bash
python scripts/process_data.py
```

#### 模型训练
```bash
python scripts/train_model.py
```

#### 年龄预测
```bash
python scripts/predict.py
```

## 配置文件说明

配置文件位于 `configs/config.yaml`，包含以下配置项：

### 数据配置
- `raw_data_path`: 原始数据路径
- `processed_data_path`: 处理后数据路径
- `test_size`: 测试集比例
- `random_state`: 随机种子

### 模型配置
- `type`: 模型类型
- `params`: 模型参数

### 特征配置
- `categorical`: 分类特征列表
- `numerical`: 数值特征列表

### 输出配置
- `model_path`: 模型保存路径
- `predictions_path`: 预测结果路径
- `metrics_path`: 评估指标路径

## 扩展建议

1. **特征优化**：
   - 探索更多属性字段的解析方法
   - 尝试其他时间特征（如季节、星期等）
   - 考虑交叉特征的构建

2. **模型调优**：
   - 使用网格搜索或贝叶斯优化进行超参数调优
   - 尝试其他集成学习方法（如XGBoost、LightGBM）
   - 考虑神经网络模型

3. **异常值处理**：
   - 进一步分析负年龄值的来源
   - 设计更合理的异常值处理策略

4. **模型验证**：
   - 实现交叉验证以更稳定地评估模型性能
   - 添加学习曲线和验证曲线分析