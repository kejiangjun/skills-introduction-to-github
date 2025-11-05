# 机器学习模型训练、微调和评估指南

## 简介

本项目提供了一个完整的机器学习训练框架，支持模型训练、微调和性能评估。适用于各种scikit-learn兼容的机器学习模型。

## 功能特性

- ✅ **模型训练**: 支持训练各种机器学习模型
- ✅ **数据预处理**: 自动进行数据标准化和归一化
- ✅ **模型微调**: 使用网格搜索自动寻找最佳超参数
- ✅ **性能评估**: 全面的评估指标（准确率、精确率、召回率、F1分数）
- ✅ **模型持久化**: 保存和加载训练好的模型
- ✅ **训练历史**: 记录和保存训练过程
- ✅ **可视化**: 丰富的图表展示训练结果和模型性能

## 安装依赖

```bash
pip install -r requirements.txt
```

## 快速开始

### 1. 基础模型训练

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from model_trainer import ModelTrainer

# 加载数据
iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# 创建训练器
model = RandomForestClassifier(n_estimators=100)
trainer = ModelTrainer(model, model_name="my_model")

# 训练模型
trainer.train(X_train, y_train, X_test, y_test)

# 评估模型
metrics = trainer.evaluate(X_test, y_test)

# 保存模型
trainer.save_model()
```

### 2. 模型微调

```python
from sklearn.linear_model import LogisticRegression
from model_trainer import ModelTrainer

# 创建模型和训练器
model = LogisticRegression()
trainer = ModelTrainer(model, model_name="tuned_model")

# 定义参数网格
param_grid = {
    'C': [0.1, 1.0, 10.0],
    'penalty': ['l1', 'l2'],
    'solver': ['liblinear']
}

# 微调模型
trainer.fine_tune(X_train, y_train, param_grid, cv=5)

# 评估微调后的模型
metrics = trainer.evaluate(X_test, y_test)
```

### 3. 模型预测

```python
# 加载已保存的模型
trainer = ModelTrainer(RandomForestClassifier(), model_name="my_model")
trainer.load_model()

# 进行预测
predictions = trainer.predict(new_data)
```

### 4. 可视化

```python
from visualization import ModelVisualizer

# 创建可视化器
visualizer = ModelVisualizer(output_dir="plots")

# 绘制混淆矩阵
visualizer.plot_confusion_matrix(y_test, y_pred, class_names=['Class 0', 'Class 1'])

# 绘制训练历史
visualizer.plot_training_history(trainer.training_history)

# 绘制模型对比
visualizer.plot_metrics_comparison(metrics_dict)
```

## 运行示例

运行完整的示例程序：

```bash
python example_usage.py
```

这将演示：
1. 基础模型训练
2. 模型微调
3. 多模型性能对比
4. 模型加载和预测

## 主要模块说明

### ModelTrainer 类

核心训练器类，提供以下方法：

- `train()`: 训练模型
- `fine_tune()`: 微调模型参数
- `evaluate()`: 评估模型性能
- `save_model()`: 保存模型
- `load_model()`: 加载模型
- `predict()`: 使用模型进行预测
- `preprocess_data()`: 数据预处理

### ModelVisualizer 类

可视化工具类，提供以下方法：

- `plot_confusion_matrix()`: 绘制混淆矩阵
- `plot_training_history()`: 绘制训练历史
- `plot_metrics_comparison()`: 绘制模型对比图
- `plot_feature_importance()`: 绘制特征重要性
- `plot_learning_curve()`: 绘制学习曲线

## 性能评估指标

模型评估提供以下指标：

1. **准确率 (Accuracy)**: 正确预测的样本比例
2. **精确率 (Precision)**: 预测为正例中真正为正例的比例
3. **召回率 (Recall)**: 真正为正例中被正确预测的比例
4. **F1分数 (F1-Score)**: 精确率和召回率的调和平均数

## 支持的模型

本框架支持所有scikit-learn兼容的模型，包括但不限于：

- 随机森林 (Random Forest)
- 逻辑回归 (Logistic Regression)
- 支持向量机 (SVM)
- 决策树 (Decision Tree)
- 梯度提升树 (Gradient Boosting)
- K近邻 (K-Nearest Neighbors)
- 朴素贝叶斯 (Naive Bayes)

## 文件结构

```
.
├── model_trainer.py      # 核心训练模块
├── visualization.py      # 可视化模块
├── example_usage.py      # 使用示例
├── requirements.txt      # 依赖包列表
├── ML_TRAINING_GUIDE.md  # 本文档
├── models/               # 保存的模型目录（自动创建）
└── plots/                # 可视化图表目录（自动创建）
```

## 最佳实践

1. **数据预处理**: 始终对数据进行标准化处理，特别是对于距离相关的算法
2. **交叉验证**: 使用交叉验证来评估模型的泛化能力
3. **参数调优**: 使用网格搜索或随机搜索来寻找最佳超参数
4. **模型保存**: 定期保存训练好的模型，避免重复训练
5. **性能监控**: 监控训练集和验证集的性能，防止过拟合

## 注意事项

- 确保数据质量和数据量充足
- 选择合适的评估指标（对于不平衡数据集，F1分数可能比准确率更合适）
- 注意模型的计算资源消耗
- 定期更新和重新训练模型以适应新数据

## 贡献

欢迎提交问题和改进建议！

## 许可证

本项目采用 MIT 许可证。
