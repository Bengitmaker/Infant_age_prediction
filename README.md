# 婴儿年龄预测项目

本项目旨在通过机器学习方法预测婴儿的年龄（以月为单位）。通过特征工程和梯度提升算法，我们能够构建一个准确的预测模型，目前达到 R² Score = 0.1673 的性能。

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

## 安装依赖

```bash
pip install -r requirements.txt
```

## 使用方法

### 1. 运行完整流水线

```bash
python scripts/run_pipeline.py
```

### 2. 单独运行各步骤

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

## 配置文件

配置文件位于 `configs/config.yaml`，可以调整以下参数：

- 数据路径
- 模型参数
- 特征配置
- 输出路径

## 特征工程

本项目通过以下方式创建特征：

1. **基础分类特征**：
   - cat_id
   - cat1
   - gender

2. **商品属性特征**：
   - property_count：属性数量
   - has_special_property：是否包含特定属性
   - sum_properties：属性值总和

3. **时间特征**：
   - day_year：购买年份
   - day_month：购买月份

4. **数值特征**：
   - buy_mount_log：购买数量的对数值
   - auction_id_last_digits：拍卖ID的后三位

## 模型性能

当前模型使用GradientBoostingRegressor，性能指标如下：

- R² Score: 0.1673
- RMSE: 21.77

## 项目模块说明

### configs/
配置管理模块，包含配置文件和配置管理器。

### scripts/
可执行脚本，包括数据处理、模型训练和预测脚本。

### utils/
工具函数模块，包括数据可视化和模型评估工具。

### models/
模型文件存储目录。

### output/
输出结果存储目录，包括预测结果和评估指标。

### docs/
详细技术文档目录。

## 技术细节

更多技术细节请参考 [技术文档](docs/technical_documentation.md)。