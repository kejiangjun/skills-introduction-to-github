"""
示例用法 (Example Usage)
演示如何使用模型训练、微调和评估功能
"""

import numpy as np
from sklearn.datasets import load_iris, make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from model_trainer import ModelTrainer
import warnings
warnings.filterwarnings('ignore')


def example_basic_training():
    """示例1: 基础模型训练"""
    print("\n" + "="*60)
    print("示例1: 基础模型训练 (Basic Model Training)")
    print("="*60)
    
    # 加载数据集
    iris = load_iris()
    X, y = iris.data, iris.target
    
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # 创建模型训练器
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    trainer = ModelTrainer(rf_model, model_name="iris_random_forest")
    
    # 训练模型
    trainer.train(X_train, y_train, X_test, y_test)
    
    # 评估模型
    metrics = trainer.evaluate(X_test, y_test)
    
    # 保存模型
    trainer.save_model()
    
    return trainer, metrics


def example_model_fine_tuning():
    """示例2: 模型微调"""
    print("\n" + "="*60)
    print("示例2: 模型微调 (Model Fine-tuning)")
    print("="*60)
    
    # 生成分类数据集
    X, y = make_classification(
        n_samples=1000, 
        n_features=20, 
        n_informative=15,
        n_redundant=5,
        random_state=42
    )
    
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # 创建逻辑回归模型
    lr_model = LogisticRegression(random_state=42, max_iter=1000)
    trainer = ModelTrainer(lr_model, model_name="tuned_logistic_regression")
    
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
    
    # 保存模型
    trainer.save_model()
    
    return trainer, metrics


def example_model_comparison():
    """示例3: 多模型性能对比"""
    print("\n" + "="*60)
    print("示例3: 多模型性能对比 (Model Comparison)")
    print("="*60)
    
    # 加载数据集
    iris = load_iris()
    X, y = iris.data, iris.target
    
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # 定义多个模型
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'SVM': SVC(kernel='rbf', random_state=42)
    }
    
    results = {}
    
    # 训练和评估每个模型
    for model_name, model in models.items():
        print(f"\n训练模型: {model_name}")
        trainer = ModelTrainer(model, model_name=model_name.lower().replace(' ', '_'))
        trainer.train(X_train, y_train)
        metrics = trainer.evaluate(X_test, y_test, detailed=False)
        results[model_name] = metrics
    
    # 输出对比结果
    print("\n" + "="*60)
    print("模型性能对比总结")
    print("="*60)
    print(f"{'模型':<25} {'准确率':<10} {'精确率':<10} {'召回率':<10} {'F1分数':<10}")
    print("-"*60)
    for model_name, metrics in results.items():
        print(f"{model_name:<25} {metrics['accuracy']:<10.4f} {metrics['precision']:<10.4f} "
              f"{metrics['recall']:<10.4f} {metrics['f1_score']:<10.4f}")
    
    return results


def example_load_and_predict():
    """示例4: 加载模型并进行预测"""
    print("\n" + "="*60)
    print("示例4: 加载模型并进行预测 (Load Model and Predict)")
    print("="*60)
    
    # 创建一个新的训练器实例
    trainer = ModelTrainer(RandomForestClassifier(), model_name="iris_random_forest")
    
    try:
        # 加载之前保存的模型
        trainer.load_model()
        
        # 创建新的测试数据
        test_samples = np.array([
            [5.1, 3.5, 1.4, 0.2],  # 预期: Setosa
            [6.5, 3.0, 5.2, 2.0],  # 预期: Virginica
            [5.9, 3.0, 4.2, 1.5],  # 预期: Versicolor
        ])
        
        # 进行预测
        predictions = trainer.predict(test_samples)
        
        # 输出预测结果
        iris = load_iris()
        class_names = iris.target_names
        
        print("\n预测结果:")
        for i, pred in enumerate(predictions):
            print(f"样本 {i+1}: {class_names[pred]}")
        
    except FileNotFoundError:
        print("错误: 未找到保存的模型。请先运行基础训练示例。")


def main():
    """主函数 - 运行所有示例"""
    print("\n" + "="*60)
    print("机器学习模型训练、微调和评估示例")
    print("Machine Learning Model Training, Fine-tuning, and Evaluation")
    print("="*60)
    
    # 运行所有示例
    example_basic_training()
    example_model_fine_tuning()
    example_model_comparison()
    example_load_and_predict()
    
    print("\n" + "="*60)
    print("所有示例运行完成！")
    print("="*60)


if __name__ == "__main__":
    main()
