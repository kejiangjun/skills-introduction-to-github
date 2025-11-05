"""
模型训练模块 (Model Training Module)
支持多种机器学习模型的训练、微调和评估
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import joblib
import json
import os
from datetime import datetime


class ModelTrainer:
    """
    模型训练器类
    支持模型训练、微调、评估和持久化
    """
    
    def __init__(self, model, model_name="model"):
        """
        初始化模型训练器
        
        Args:
            model: sklearn兼容的模型对象
            model_name: 模型名称，用于保存
        """
        self.model = model
        self.model_name = model_name
        self.scaler = StandardScaler()
        self.training_history = {
            'train_scores': [],
            'val_scores': [],
            'timestamps': []
        }
        self.best_params = None
        
    def preprocess_data(self, X, fit=True):
        """
        数据预处理和标准化
        
        Args:
            X: 特征数据
            fit: 是否拟合标准化器（训练集为True，测试集为False）
            
        Returns:
            标准化后的数据
        """
        if fit:
            return self.scaler.fit_transform(X)
        else:
            return self.scaler.transform(X)
    
    def train(self, X_train, y_train, X_val=None, y_val=None, preprocess=True):
        """
        训练模型
        
        Args:
            X_train: 训练特征
            y_train: 训练标签
            X_val: 验证特征（可选）
            y_val: 验证标签（可选）
            preprocess: 是否进行数据预处理
            
        Returns:
            训练后的模型
        """
        print(f"开始训练模型: {self.model_name}")
        print(f"训练样本数: {len(X_train)}")
        
        # 数据预处理
        if preprocess:
            X_train_processed = self.preprocess_data(X_train, fit=True)
            if X_val is not None:
                X_val_processed = self.preprocess_data(X_val, fit=False)
        else:
            X_train_processed = X_train
            X_val_processed = X_val
        
        # 训练模型
        start_time = datetime.now()
        self.model.fit(X_train_processed, y_train)
        end_time = datetime.now()
        training_time = (end_time - start_time).total_seconds()
        
        # 计算训练集得分
        train_score = self.model.score(X_train_processed, y_train)
        self.training_history['train_scores'].append(train_score)
        self.training_history['timestamps'].append(datetime.now().isoformat())
        
        print(f"训练完成！用时: {training_time:.2f}秒")
        print(f"训练集准确率: {train_score:.4f}")
        
        # 如果提供了验证集，计算验证集得分
        if X_val is not None and y_val is not None:
            val_score = self.model.score(X_val_processed, y_val)
            self.training_history['val_scores'].append(val_score)
            print(f"验证集准确率: {val_score:.4f}")
        
        return self.model
    
    def fine_tune(self, X_train, y_train, param_grid, cv=5, scoring='accuracy'):
        """
        模型微调 - 使用网格搜索寻找最佳参数
        
        Args:
            X_train: 训练特征
            y_train: 训练标签
            param_grid: 参数网格字典
            cv: 交叉验证折数
            scoring: 评分指标
            
        Returns:
            微调后的最佳模型
        """
        print(f"开始微调模型: {self.model_name}")
        print(f"参数网格: {param_grid}")
        print(f"交叉验证折数: {cv}")
        
        # 数据预处理
        X_train_processed = self.preprocess_data(X_train, fit=True)
        
        # 网格搜索
        grid_search = GridSearchCV(
            self.model, 
            param_grid, 
            cv=cv, 
            scoring=scoring,
            n_jobs=-1,
            verbose=1
        )
        
        start_time = datetime.now()
        grid_search.fit(X_train_processed, y_train)
        end_time = datetime.now()
        tuning_time = (end_time - start_time).total_seconds()
        
        # 保存最佳参数
        self.best_params = grid_search.best_params_
        self.model = grid_search.best_estimator_
        
        print(f"微调完成！用时: {tuning_time:.2f}秒")
        print(f"最佳参数: {self.best_params}")
        print(f"最佳交叉验证得分: {grid_search.best_score_:.4f}")
        
        return self.model
    
    def evaluate(self, X_test, y_test, detailed=True):
        """
        模型性能评估
        
        Args:
            X_test: 测试特征
            y_test: 测试标签
            detailed: 是否输出详细评估报告
            
        Returns:
            评估指标字典
        """
        print(f"\n{'='*50}")
        print(f"模型性能评估: {self.model_name}")
        print(f"{'='*50}")
        
        # 数据预处理
        X_test_processed = self.preprocess_data(X_test, fit=False)
        
        # 预测
        y_pred = self.model.predict(X_test_processed)
        
        # 计算评估指标
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
            'f1_score': f1_score(y_test, y_pred, average='weighted', zero_division=0)
        }
        
        # 输出评估结果
        print(f"准确率 (Accuracy):  {metrics['accuracy']:.4f}")
        print(f"精确率 (Precision): {metrics['precision']:.4f}")
        print(f"召回率 (Recall):    {metrics['recall']:.4f}")
        print(f"F1分数 (F1-Score):  {metrics['f1_score']:.4f}")
        
        if detailed:
            print(f"\n混淆矩阵 (Confusion Matrix):")
            cm = confusion_matrix(y_test, y_pred)
            print(cm)
        
        print(f"{'='*50}\n")
        
        return metrics
    
    def save_model(self, directory="models", save_history=True):
        """
        保存模型和训练历史
        
        Args:
            directory: 保存目录
            save_history: 是否保存训练历史
        """
        # 创建目录
        os.makedirs(directory, exist_ok=True)
        
        # 保存模型
        model_path = os.path.join(directory, f"{self.model_name}.joblib")
        joblib.dump(self.model, model_path)
        print(f"模型已保存至: {model_path}")
        
        # 保存标准化器
        scaler_path = os.path.join(directory, f"{self.model_name}_scaler.joblib")
        joblib.dump(self.scaler, scaler_path)
        print(f"标准化器已保存至: {scaler_path}")
        
        # 保存训练历史和参数
        if save_history:
            history_path = os.path.join(directory, f"{self.model_name}_history.json")
            history_data = {
                'training_history': self.training_history,
                'best_params': self.best_params,
                'model_name': self.model_name,
                'saved_at': datetime.now().isoformat()
            }
            with open(history_path, 'w', encoding='utf-8') as f:
                json.dump(history_data, f, indent=4, ensure_ascii=False)
            print(f"训练历史已保存至: {history_path}")
    
    def load_model(self, directory="models"):
        """
        加载模型和标准化器
        
        Args:
            directory: 模型所在目录
        """
        # 加载模型
        model_path = os.path.join(directory, f"{self.model_name}.joblib")
        self.model = joblib.load(model_path)
        print(f"模型已加载: {model_path}")
        
        # 加载标准化器
        scaler_path = os.path.join(directory, f"{self.model_name}_scaler.joblib")
        self.scaler = joblib.load(scaler_path)
        print(f"标准化器已加载: {scaler_path}")
        
        # 加载训练历史
        history_path = os.path.join(directory, f"{self.model_name}_history.json")
        if os.path.exists(history_path):
            with open(history_path, 'r', encoding='utf-8') as f:
                history_data = json.load(f)
                self.training_history = history_data.get('training_history', {})
                self.best_params = history_data.get('best_params')
            print(f"训练历史已加载: {history_path}")
    
    def predict(self, X, preprocess=True):
        """
        使用模型进行预测
        
        Args:
            X: 输入特征
            preprocess: 是否进行数据预处理
            
        Returns:
            预测结果
        """
        if preprocess:
            X_processed = self.preprocess_data(X, fit=False)
        else:
            X_processed = X
        
        return self.model.predict(X_processed)
