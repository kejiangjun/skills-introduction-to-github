"""
可视化示例 (Visualization Example)
演示如何使用可视化功能展示模型性能
"""

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from model_trainer import ModelTrainer
from visualization import ModelVisualizer
import warnings
warnings.filterwarnings('ignore')


def visualize_single_model():
    """示例1: 单个模型的可视化"""
    print("\n" + "="*60)
    print("示例1: 单个模型可视化 (Single Model Visualization)")
    print("="*60)
    
    # 加载数据
    iris = load_iris()
    X, y = iris.data, iris.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # 训练模型
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    trainer = ModelTrainer(model, model_name="visualization_rf")
    trainer.train(X_train, y_train, X_test, y_test)
    
    # 获取预测结果
    X_test_processed = trainer.preprocess_data(X_test, fit=False)
    y_pred = trainer.model.predict(X_test_processed)
    
    # 创建可视化器
    visualizer = ModelVisualizer(output_dir="plots")
    
    # 绘制混淆矩阵
    visualizer.plot_confusion_matrix(
        y_test, y_pred, 
        class_names=iris.target_names,
        title="Random Forest - Confusion Matrix",
        save_name="rf_confusion_matrix.png"
    )
    
    # 绘制训练历史
    visualizer.plot_training_history(
        trainer.training_history,
        title="Random Forest - Training History",
        save_name="rf_training_history.png"
    )
    
    # 绘制特征重要性
    if hasattr(trainer.model, 'feature_importances_'):
        visualizer.plot_feature_importance(
            trainer.model.feature_importances_,
            feature_names=iris.feature_names,
            title="Random Forest - Feature Importance",
            save_name="rf_feature_importance.png"
        )
    
    print("\n可视化完成！图表已保存至 'plots/' 目录")


def visualize_model_comparison():
    """示例2: 多模型对比可视化"""
    print("\n" + "="*60)
    print("示例2: 多模型对比可视化 (Model Comparison Visualization)")
    print("="*60)
    
    # 加载数据
    iris = load_iris()
    X, y = iris.data, iris.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # 训练多个模型
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'SVM': SVC(kernel='rbf', random_state=42)
    }
    
    metrics_dict = {}
    
    for model_name, model in models.items():
        trainer = ModelTrainer(model, model_name=model_name)
        trainer.train(X_train, y_train)
        metrics = trainer.evaluate(X_test, y_test, detailed=False)
        metrics_dict[model_name] = metrics
    
    # 创建可视化器
    visualizer = ModelVisualizer(output_dir="plots")
    
    # 绘制模型对比图
    visualizer.plot_metrics_comparison(
        metrics_dict,
        title="Model Performance Comparison",
        save_name="model_comparison.png"
    )
    
    print("\n模型对比可视化完成！")


def visualize_learning_curve():
    """示例3: 学习曲线可视化"""
    print("\n" + "="*60)
    print("示例3: 学习曲线可视化 (Learning Curve Visualization)")
    print("="*60)
    
    # 加载数据
    iris = load_iris()
    X, y = iris.data, iris.target
    
    # 创建模型
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    
    # 计算学习曲线
    print("计算学习曲线...")
    train_sizes, train_scores, val_scores = learning_curve(
        model, X, y, 
        cv=5,
        train_sizes=np.linspace(0.1, 1.0, 10),
        n_jobs=-1,
        random_state=42
    )
    
    # 创建可视化器
    visualizer = ModelVisualizer(output_dir="plots")
    
    # 绘制学习曲线
    visualizer.plot_learning_curve(
        train_sizes, train_scores, val_scores,
        title="Random Forest - Learning Curve",
        save_name="learning_curve.png"
    )
    
    print("\n学习曲线可视化完成！")


def visualize_all_models_confusion_matrices():
    """示例4: 所有模型的混淆矩阵对比"""
    print("\n" + "="*60)
    print("示例4: 多个模型混淆矩阵对比")
    print("="*60)
    
    # 加载数据
    iris = load_iris()
    X, y = iris.data, iris.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # 训练多个模型
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'SVM': SVC(kernel='rbf', random_state=42)
    }
    
    # 创建可视化器
    visualizer = ModelVisualizer(output_dir="plots")
    
    for model_name, model in models.items():
        print(f"\n训练和可视化: {model_name}")
        trainer = ModelTrainer(model, model_name=model_name)
        trainer.train(X_train, y_train)
        
        # 获取预测结果
        X_test_processed = trainer.preprocess_data(X_test, fit=False)
        y_pred = trainer.model.predict(X_test_processed)
        
        # 绘制混淆矩阵
        safe_name = model_name.lower().replace(' ', '_')
        visualizer.plot_confusion_matrix(
            y_test, y_pred,
            class_names=iris.target_names,
            title=f"{model_name} - Confusion Matrix",
            save_name=f"{safe_name}_cm.png"
        )
    
    print("\n所有混淆矩阵已生成！")


def main():
    """主函数 - 运行所有可视化示例"""
    print("\n" + "="*60)
    print("机器学习可视化示例")
    print("Machine Learning Visualization Examples")
    print("="*60)
    
    # 运行所有示例
    visualize_single_model()
    visualize_model_comparison()
    visualize_learning_curve()
    visualize_all_models_confusion_matrices()
    
    print("\n" + "="*60)
    print("所有可视化示例运行完成！")
    print("请查看 'plots/' 目录中的图表")
    print("="*60)


if __name__ == "__main__":
    main()
