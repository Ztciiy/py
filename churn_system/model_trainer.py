"""
=============================================================
模块: model_trainer.py
功能: 特征工程 + 多模型训练 + 模型评估 + 模型保存
作者: 大四毕业设计项目
说明: 支持逻辑回归、随机森林、XGBoost、LightGBM 四种模型，
      并通过交叉验证与超参数搜索选出最优模型。
=============================================================
"""

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report,
    confusion_matrix, roc_curve
)
from sklearn.model_selection import StratifiedKFold, GridSearchCV
import warnings
warnings.filterwarnings('ignore')

# 尝试导入可选依赖
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("[警告] XGBoost 未安装，将跳过该模型。pip install xgboost")

try:
    from lightgbm import LGBMClassifier
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("[警告] LightGBM 未安装，将跳过该模型。pip install lightgbm")


# ──────────────────────────────────────────────
# 1. 模型定义工厂
# ──────────────────────────────────────────────

def get_model_zoo() -> dict:
    """
    返回所有候选模型字典

    返回:
        dict: {模型名称: 模型实例}
    """
    models = {
        'LogisticRegression': LogisticRegression(
            max_iter=1000,
            class_weight='balanced',
            random_state=42
        ),
        'RandomForest': RandomForestClassifier(
            n_estimators=200,
            max_depth=8,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        ),
        'GradientBoosting': GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=4,
            random_state=42
        ),
    }

    if XGBOOST_AVAILABLE:
        models['XGBoost'] = XGBClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=5,
            use_label_encoder=False,
            eval_metric='logloss',
            random_state=42,
            n_jobs=-1
        )

    if LIGHTGBM_AVAILABLE:
        models['LightGBM'] = LGBMClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=5,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1,
            verbose=-1
        )

    return models


# ──────────────────────────────────────────────
# 2. 单模型训练与评估
# ──────────────────────────────────────────────

def evaluate_model(model, X_test, y_test, model_name: str = '') -> dict:
    """
    评估模型性能，返回各项指标

    参数:
        model     : 已训练模型
        X_test    : 测试集特征
        y_test    : 测试集标签
        model_name: 模型名称（用于打印）

    返回:
        dict: 包含 accuracy, precision, recall, f1, auc 等指标
    """
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        'model'    : model_name,
        'accuracy' : accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall'   : recall_score(y_test, y_pred, zero_division=0),
        'f1'       : f1_score(y_test, y_pred, zero_division=0),
        'auc'      : roc_auc_score(y_test, y_proba),
    }

    print(f"\n{'='*55}")
    print(f"  模型: {model_name}")
    print(f"{'='*55}")
    print(f"  准确率  (Accuracy) : {metrics['accuracy']:.4f}")
    print(f"  精确率  (Precision): {metrics['precision']:.4f}")
    print(f"  召回率  (Recall)   : {metrics['recall']:.4f}")
    print(f"  F1 分数 (F1-Score) : {metrics['f1']:.4f}")
    print(f"  AUC-ROC            : {metrics['auc']:.4f}")
    print(f"\n  分类报告:\n{classification_report(y_test, y_pred, target_names=['留存','流失'])}")

    return metrics


# ──────────────────────────────────────────────
# 3. 交叉验证训练
# ──────────────────────────────────────────────

def cross_validate_models(models: dict, X_train, y_train, cv: int = 5) -> pd.DataFrame:
    """
    对所有模型进行 K 折交叉验证，比较稳定性

    参数:
        models : 模型字典
        X_train: 训练集特征
        y_train: 训练集标签
        cv     : 折数，默认 5

    返回:
        DataFrame: 每个模型的交叉验证均值与标准差
    """
    print(f"\n[交叉验证] {cv} 折分层交叉验证中...")
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    results = []

    for name, model in models.items():
        fold_aucs = []
        fold_f1s  = []

        for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
            X_tr, X_vl = X_train[train_idx], X_train[val_idx]
            y_tr, y_vl = y_train[train_idx], y_train[val_idx]

            model.fit(X_tr, y_tr)
            y_proba = model.predict_proba(X_vl)[:, 1]
            y_pred  = model.predict(X_vl)

            fold_aucs.append(roc_auc_score(y_vl, y_proba))
            fold_f1s.append(f1_score(y_vl, y_pred, zero_division=0))

        results.append({
            'model'       : name,
            'cv_auc_mean' : np.mean(fold_aucs),
            'cv_auc_std'  : np.std(fold_aucs),
            'cv_f1_mean'  : np.mean(fold_f1s),
            'cv_f1_std'   : np.std(fold_f1s),
        })
        print(f"  {name:25s} AUC={np.mean(fold_aucs):.4f}±{np.std(fold_aucs):.4f}  "
              f"F1={np.mean(fold_f1s):.4f}±{np.std(fold_f1s):.4f}")

    return pd.DataFrame(results).sort_values('cv_auc_mean', ascending=False)


# ──────────────────────────────────────────────
# 4. 超参数搜索（以随机森林为例）
# ──────────────────────────────────────────────

def tune_random_forest(X_train, y_train) -> RandomForestClassifier:
    """
    对随机森林进行网格搜索调参

    参数:
        X_train: 训练集特征
        y_train: 训练集标签

    返回:
        最优随机森林模型
    """
    print("\n[调参] 随机森林网格搜索...")
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth'   : [6, 8, 10],
        'min_samples_split': [2, 5],
    }
    rf = RandomForestClassifier(class_weight='balanced', random_state=42, n_jobs=-1)
    gs = GridSearchCV(
        rf, param_grid, cv=3, scoring='roc_auc',
        n_jobs=-1, verbose=0, refit=True
    )
    gs.fit(X_train, y_train)
    print(f"[调参] 最优参数: {gs.best_params_}，最优 AUC={gs.best_score_:.4f}")
    return gs.best_estimator_


# ──────────────────────────────────────────────
# 5. 特征重要性分析
# ──────────────────────────────────────────────

def get_feature_importance(model, feature_names: list) -> pd.DataFrame:
    """
    提取基于树模型的特征重要性

    参数:
        model        : 已训练的树模型
        feature_names: 特征名列表

    返回:
        DataFrame: 特征名 + 重要性分数（降序排列）
    """
    if hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importance = np.abs(model.coef_[0])
    else:
        print("[警告] 该模型不支持特征重要性提取")
        return pd.DataFrame()

    df = pd.DataFrame({
        'feature'   : feature_names,
        'importance': importance
    }).sort_values('importance', ascending=False).reset_index(drop=True)

    print("\n[特征重要性 Top 10]")
    print(df.head(10).to_string(index=False))
    return df


# ──────────────────────────────────────────────
# 6. 模型持久化
# ──────────────────────────────────────────────

def save_model(model, path: str, model_name: str = 'model'):
    """
    保存模型到本地文件

    参数:
        model     : 训练好的模型
        path      : 保存目录
        model_name: 文件名（不含扩展名）
    """
    os.makedirs(path, exist_ok=True)
    file_path = os.path.join(path, f'{model_name}.pkl')
    with open(file_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"[保存] 模型已保存至: {file_path}")


def load_model(path: str):
    """
    从本地文件加载模型

    参数:
        path: .pkl 文件路径

    返回:
        模型对象
    """
    with open(path, 'rb') as f:
        model = pickle.load(f)
    print(f"[加载] 模型已从 {path} 加载")
    return model


# ──────────────────────────────────────────────
# 7. 完整训练流程入口
# ──────────────────────────────────────────────

def train_all_models(X_train, y_train, X_test, y_test,
                     feature_names: list, model_save_dir: str = 'models'):
    """
    训练所有模型，交叉验证对比，选出最优模型并保存

    参数:
        X_train        : 训练集特征
        y_train        : 训练集标签
        X_test         : 测试集特征
        y_test         : 测试集标签
        feature_names  : 特征名列表
        model_save_dir : 模型保存目录

    返回:
        best_model    : 最优模型
        all_metrics   : 所有模型评估指标列表
        feature_imp_df: 特征重要性 DataFrame
        cv_results    : 交叉验证结果 DataFrame
    """
    models = get_model_zoo()

    # 交叉验证比较
    cv_results = cross_validate_models(models, X_train, y_train)
    best_model_name = cv_results.iloc[0]['model']
    print(f"\n[选模型] 交叉验证最优模型: {best_model_name}")

    # 全量训练所有模型并在测试集评估
    all_metrics = []
    trained_models = {}

    for name, model in models.items():
        print(f"\n[训练] 正在训练 {name}...")
        model.fit(X_train, y_train)
        metrics = evaluate_model(model, X_test, y_test, model_name=name)
        all_metrics.append(metrics)
        trained_models[name] = model
        save_model(model, model_save_dir, name)

    # 选出 AUC 最高的模型
    best_model = trained_models[
        max(all_metrics, key=lambda x: x['auc'])['model']
    ]

    # 特征重要性
    feature_imp_df = get_feature_importance(best_model, feature_names)

    return best_model, all_metrics, feature_imp_df, cv_results
