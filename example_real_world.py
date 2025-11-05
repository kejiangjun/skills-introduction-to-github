"""
真实场景示例 (Real-world Example)
展示如何在实际项目中使用本框架进行完整的机器学习工作流程
"""

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from model_trainer import ModelTrainer
from visualization import ModelVisualizer
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')


def generate_sample_dataset():
    """
    生成模拟的客户流失预测数据集
    在实际应用中，这里应该从数据库或文件加载真实数据
    """
    print("生成模拟数据集...")
    X, y = make_classification(
        n_samples=2000,
        n_features=20,
        n_informative=15,
        n_redundant=3,
        n_classes=2,
        weights=[0.7, 0.3],  # 模拟不平衡数据集
        random_state=42
    )
    
    # 创建特征名称
    feature_names = [
        'age', 'tenure', 'monthly_charges', 'total_charges', 'num_products',
        'num_interactions', 'satisfaction_score', 'complaint_count', 'usage_freq',
        'contract_type', 'payment_method', 'paperless_billing', 'tech_support',
        'online_security', 'device_protection', 'backup_service', 'streaming_tv',
        'streaming_movies', 'internet_service', 'phone_service'
    ]
    
    # 转换为 DataFrame
    df = pd.DataFrame(X, columns=feature_names[:X.shape[1]])
    df['churn'] = y
    
    print(f"数据集大小: {len(df)} 样本")
    print(f"特征数量: {X.shape[1]}")
    print(f"流失客户比例: {y.mean():.2%}")
    
    return df, feature_names[:X.shape[1]]


def complete_ml_workflow():
    """
    完整的机器学习工作流程示例
    """
    print("\n" + "="*70)
    print("真实场景：客户流失预测完整工作流程")
    print("Real-world Example: Customer Churn Prediction Workflow")
    print("="*70)
    
    # 步骤1: 数据准备
    print("\n【步骤 1/6】数据准备")
    print("-" * 70)
    df, feature_names = generate_sample_dataset()
    
    # 分离特征和标签
    X = df.drop('churn', axis=1).values
    y = df['churn'].values
    
    # 划分训练集、验证集和测试集
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.2, random_state=42, stratify=y_temp
    )
    
    print(f"训练集: {len(X_train)} 样本")
    print(f"验证集: {len(X_val)} 样本")
    print(f"测试集: {len(X_test)} 样本")
    
    # 步骤2: 模型选择和初始训练
    print("\n【步骤 2/6】模型选择和初始训练")
    print("-" * 70)
    
    # 定义候选模型
    candidate_models = {
        'Random Forest': RandomForestClassifier(random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42),
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'SVM': SVC(kernel='rbf', random_state=42)
    }
    
    # 训练所有候选模型
    trained_models = {}
    initial_metrics = {}
    
    for model_name, model in candidate_models.items():
        print(f"\n训练 {model_name}...")
        trainer = ModelTrainer(model, model_name=model_name.lower().replace(' ', '_'))
        trainer.train(X_train, y_train, X_val, y_val)
        metrics = trainer.evaluate(X_val, y_val, detailed=False)
        
        trained_models[model_name] = trainer
        initial_metrics[model_name] = metrics
    
    # 步骤3: 模型对比和选择
    print("\n【步骤 3/6】模型性能对比")
    print("-" * 70)
    print(f"{'模型':<25} {'准确率':<12} {'精确率':<12} {'召回率':<12} {'F1分数':<12}")
    print("-" * 70)
    
    best_model_name = None
    best_f1_score = 0
    
    for model_name, metrics in initial_metrics.items():
        print(f"{model_name:<25} {metrics['accuracy']:<12.4f} "
              f"{metrics['precision']:<12.4f} {metrics['recall']:<12.4f} "
              f"{metrics['f1_score']:<12.4f}")
        
        if metrics['f1_score'] > best_f1_score:
            best_f1_score = metrics['f1_score']
            best_model_name = model_name
    
    print(f"\n最佳模型: {best_model_name} (F1分数: {best_f1_score:.4f})")
    best_trainer = trained_models[best_model_name]
    
    # 步骤4: 最佳模型微调
    print(f"\n【步骤 4/6】微调最佳模型 ({best_model_name})")
    print("-" * 70)
    
    # 为最佳模型定义参数网格
    if best_model_name == 'Random Forest':
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5]
        }
    elif best_model_name == 'Gradient Boosting':
        param_grid = {
            'n_estimators': [50, 100],
            'learning_rate': [0.01, 0.1],
            'max_depth': [3, 5]
        }
    elif best_model_name == 'Logistic Regression':
        param_grid = {
            'C': [0.1, 1.0, 10.0],
            'penalty': ['l1', 'l2'],
            'solver': ['liblinear']
        }
    else:  # SVM
        param_grid = {
            'C': [0.1, 1, 10],
            'gamma': ['scale', 'auto']
        }
    
    # 微调模型
    tuned_trainer = ModelTrainer(
        candidate_models[best_model_name],
        model_name=f"tuned_{best_model_name.lower().replace(' ', '_')}"
    )
    tuned_trainer.fine_tune(X_train, y_train, param_grid, cv=3, scoring='f1')
    
    # 步骤5: 最终评估
    print("\n【步骤 5/6】在测试集上进行最终评估")
    print("-" * 70)
    final_metrics = tuned_trainer.evaluate(X_test, y_test, detailed=True)
    
    # 步骤6: 模型保存和可视化
    print("\n【步骤 6/6】保存模型和生成可视化报告")
    print("-" * 70)
    
    # 保存最佳模型
    tuned_trainer.save_model(directory="models")
    
    # 创建可视化
    visualizer = ModelVisualizer(output_dir="plots")
    
    # 对比所有模型
    visualizer.plot_metrics_comparison(
        initial_metrics,
        title="Model Comparison - Churn Prediction",
        save_name="churn_model_comparison.png"
    )
    
    # 最佳模型的混淆矩阵
    X_test_processed = tuned_trainer.preprocess_data(X_test, fit=False)
    y_pred = tuned_trainer.model.predict(X_test_processed)
    
    visualizer.plot_confusion_matrix(
        y_test, y_pred,
        class_names=['Not Churned', 'Churned'],
        title=f"{best_model_name} - Churn Prediction",
        save_name="churn_confusion_matrix.png"
    )
    
    # 特征重要性（如果模型支持）
    if hasattr(tuned_trainer.model, 'feature_importances_'):
        visualizer.plot_feature_importance(
            tuned_trainer.model.feature_importances_,
            feature_names=feature_names,
            title=f"{best_model_name} - Feature Importance",
            save_name="churn_feature_importance.png",
            top_n=15
        )
        print("✓ 特征重要性图已生成")
    
    print("✓ 模型已保存")
    print("✓ 可视化报告已生成")
    
    # 最终总结
    print("\n" + "="*70)
    print("工作流程完成总结")
    print("="*70)
    print(f"最佳模型: {best_model_name}")
    print(f"最佳参数: {tuned_trainer.best_params}")
    print(f"\n测试集性能:")
    print(f"  - 准确率: {final_metrics['accuracy']:.4f}")
    print(f"  - 精确率: {final_metrics['precision']:.4f}")
    print(f"  - 召回率: {final_metrics['recall']:.4f}")
    print(f"  - F1分数: {final_metrics['f1_score']:.4f}")
    print(f"\n模型文件: models/tuned_{best_model_name.lower().replace(' ', '_')}.joblib")
    print(f"可视化报告: plots/ 目录")
    print("="*70)
    
    return tuned_trainer, final_metrics


def demonstrate_model_usage(trainer):
    """
    演示如何使用训练好的模型进行预测
    """
    print("\n" + "="*70)
    print("模型应用示例：对新客户进行流失预测")
    print("="*70)
    
    # 模拟新客户数据
    new_customers = np.array([
        # 高风险客户：低满意度，高投诉
        [-0.5, -1.0, 0.8, 1.2, -0.8, -1.5, -2.0, 2.5, -0.6, 0.5,
         0.3, -0.4, -1.0, -0.8, -0.7, -0.9, 0.2, 0.1, 0.4, -0.5],
        # 低风险客户：高满意度，低投诉
        [0.5, 1.2, -0.3, -0.5, 1.0, 1.2, 1.8, -1.0, 1.2, -0.3,
         -0.2, 0.5, 0.8, 0.9, 0.7, 0.8, -0.1, -0.2, -0.3, 0.6]
    ])
    
    # 进行预测
    predictions = trainer.predict(new_customers)
    
    # 如果模型支持概率预测
    if hasattr(trainer.model, 'predict_proba'):
        X_processed = trainer.preprocess_data(new_customers, fit=False)
        probabilities = trainer.model.predict_proba(X_processed)
        
        print("\n预测结果:")
        for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
            churn_prob = prob[1] * 100
            status = "会流失" if pred == 1 else "不会流失"
            risk = "高风险" if churn_prob > 70 else "中风险" if churn_prob > 40 else "低风险"
            
            print(f"\n客户 {i+1}:")
            print(f"  预测结果: {status}")
            print(f"  流失概率: {churn_prob:.2f}%")
            print(f"  风险等级: {risk}")
            
            if pred == 1:
                print(f"  建议: 立即采取挽留措施（优惠、关怀等）")
    else:
        print("\n预测结果:")
        for i, pred in enumerate(predictions):
            status = "会流失" if pred == 1 else "不会流失"
            print(f"客户 {i+1}: {status}")


def main():
    """
    主函数：运行完整的真实场景示例
    """
    print("\n" + "="*70)
    print("机器学习完整工作流程 - 真实场景演示")
    print("Complete ML Workflow - Real-world Demonstration")
    print("="*70)
    
    # 运行完整工作流程
    trainer, metrics = complete_ml_workflow()
    
    # 演示模型应用
    demonstrate_model_usage(trainer)
    
    print("\n" + "="*70)
    print("真实场景示例运行完成！")
    print("\n这个示例展示了如何在实际项目中:")
    print("1. 准备和划分数据集")
    print("2. 训练多个候选模型")
    print("3. 对比和选择最佳模型")
    print("4. 微调模型超参数")
    print("5. 在测试集上评估性能")
    print("6. 保存模型和生成可视化报告")
    print("7. 使用模型进行实际预测")
    print("="*70)


if __name__ == "__main__":
    main()
