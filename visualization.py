"""
可视化模块 (Visualization Module)
提供训练过程和模型性能的可视化功能
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix
import os


class ModelVisualizer:
    """模型可视化工具类"""
    
    def __init__(self, output_dir="plots"):
        """
        初始化可视化器
        
        Args:
            output_dir: 图表保存目录
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # 设置样式
        sns.set_style("whitegrid")
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
    
    def plot_confusion_matrix(self, y_true, y_pred, class_names=None, 
                             title="Confusion Matrix", save_name="confusion_matrix.png"):
        """
        绘制混淆矩阵
        
        Args:
            y_true: 真实标签
            y_pred: 预测标签
            class_names: 类别名称列表
            title: 图表标题
            save_name: 保存文件名
        """
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names)
        plt.title(title, fontsize=16, fontweight='bold')
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.tight_layout()
        
        save_path = os.path.join(self.output_dir, save_name)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"混淆矩阵已保存至: {save_path}")
        plt.close()
    
    def plot_training_history(self, history, title="Training History", 
                             save_name="training_history.png"):
        """
        绘制训练历史曲线
        
        Args:
            history: 训练历史字典，包含train_scores和val_scores
            title: 图表标题
            save_name: 保存文件名
        """
        train_scores = history.get('train_scores', [])
        val_scores = history.get('val_scores', [])
        
        if not train_scores:
            print("警告: 没有训练历史数据可绘制")
            return
        
        epochs = range(1, len(train_scores) + 1)
        
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, train_scores, 'b-o', label='Training Score', linewidth=2)
        if val_scores:
            plt.plot(epochs, val_scores, 'r-s', label='Validation Score', linewidth=2)
        
        plt.title(title, fontsize=16, fontweight='bold')
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Score', fontsize=12)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        save_path = os.path.join(self.output_dir, save_name)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"训练历史图已保存至: {save_path}")
        plt.close()
    
    def plot_metrics_comparison(self, metrics_dict, title="Model Metrics Comparison",
                               save_name="metrics_comparison.png"):
        """
        绘制多个模型的性能指标对比
        
        Args:
            metrics_dict: 模型指标字典 {model_name: {metric_name: value}}
            title: 图表标题
            save_name: 保存文件名
        """
        models = list(metrics_dict.keys())
        metric_names = ['accuracy', 'precision', 'recall', 'f1_score']
        metric_labels = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        
        x = np.arange(len(models))
        width = 0.2
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        for i, (metric, label) in enumerate(zip(metric_names, metric_labels)):
            values = [metrics_dict[model].get(metric, 0) for model in models]
            ax.bar(x + i*width, values, width, label=label)
        
        ax.set_xlabel('Models', fontsize=12, fontweight='bold')
        ax.set_ylabel('Score', fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.set_xticks(x + width * 1.5)
        ax.set_xticklabels(models, rotation=45, ha='right')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim(0, 1.1)
        
        plt.tight_layout()
        save_path = os.path.join(self.output_dir, save_name)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"指标对比图已保存至: {save_path}")
        plt.close()
    
    def plot_feature_importance(self, feature_importance, feature_names=None,
                               title="Feature Importance", save_name="feature_importance.png",
                               top_n=20):
        """
        绘制特征重要性图
        
        Args:
            feature_importance: 特征重要性数组
            feature_names: 特征名称列表
            title: 图表标题
            save_name: 保存文件名
            top_n: 显示前N个重要特征
        """
        if feature_names is None:
            feature_names = [f"Feature {i}" for i in range(len(feature_importance))]
        
        # 按重要性排序
        indices = np.argsort(feature_importance)[::-1][:top_n]
        sorted_importance = feature_importance[indices]
        sorted_names = [feature_names[i] for i in indices]
        
        plt.figure(figsize=(10, max(6, top_n * 0.3)))
        colors = plt.cm.viridis(np.linspace(0, 1, len(sorted_importance)))
        plt.barh(range(len(sorted_importance)), sorted_importance, color=colors)
        plt.yticks(range(len(sorted_importance)), sorted_names)
        plt.xlabel('Importance', fontsize=12, fontweight='bold')
        plt.title(title, fontsize=16, fontweight='bold')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        
        save_path = os.path.join(self.output_dir, save_name)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"特征重要性图已保存至: {save_path}")
        plt.close()
    
    def plot_learning_curve(self, train_sizes, train_scores, val_scores,
                           title="Learning Curve", save_name="learning_curve.png"):
        """
        绘制学习曲线
        
        Args:
            train_sizes: 训练集大小数组
            train_scores: 训练得分数组
            val_scores: 验证得分数组
            title: 图表标题
            save_name: 保存文件名
        """
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        val_mean = np.mean(val_scores, axis=1)
        val_std = np.std(val_scores, axis=1)
        
        plt.figure(figsize=(10, 6))
        plt.plot(train_sizes, train_mean, 'o-', color='b', label='Training score')
        plt.plot(train_sizes, val_mean, 'o-', color='r', label='Validation score')
        
        plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, 
                        alpha=0.1, color='b')
        plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, 
                        alpha=0.1, color='r')
        
        plt.xlabel('Training Set Size', fontsize=12, fontweight='bold')
        plt.ylabel('Score', fontsize=12, fontweight='bold')
        plt.title(title, fontsize=16, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        save_path = os.path.join(self.output_dir, save_name)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"学习曲线图已保存至: {save_path}")
        plt.close()
