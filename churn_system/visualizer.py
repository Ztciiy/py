"""
=============================================================
模块: visualizer.py
功能: 数据可视化模块
作者: 大四毕业设计项目
说明: 提供 EDA 探索图、模型性能图、预警统计图等全套可视化
=============================================================
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')           # 无头服务器可用；如需弹窗可改为 TkAgg
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import rcParams
from sklearn.metrics import roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay
import warnings
warnings.filterwarnings('ignore')

# ── 中文字体设置（Windows）──
rcParams['font.family']       = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
rcParams['axes.unicode_minus'] = False
rcParams['figure.dpi']         = 120

OUTPUT_DIR = 'output'
os.makedirs(OUTPUT_DIR, exist_ok=True)


def _save(fig, filename: str):
    """统一保存图片"""
    path = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f"[图表] 已保存: {path}")
    return path


# ──────────────────────────────────────────────
# 1. EDA：数据分布与流失率探索
# ──────────────────────────────────────────────

def plot_churn_distribution(df: pd.DataFrame):
    """客户流失比例饼图 + 柱状图"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('客户流失分布总览', fontsize=16, fontweight='bold')

    churn_counts = df['Churn'].value_counts()
    colors = ['#2ecc71', '#e74c3c']

    # 饼图
    axes[0].pie(
        churn_counts, labels=['留存 (No)', '流失 (Yes)'],
        colors=colors, autopct='%1.1f%%', startangle=90,
        textprops={'fontsize': 12}
    )
    axes[0].set_title('流失比例', fontsize=13)

    # 柱状图
    bars = axes[1].bar(
        ['留存 (No)', '流失 (Yes)'], churn_counts,
        color=colors, edgecolor='white', linewidth=1.5
    )
    for bar, count in zip(bars, churn_counts):
        axes[1].text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 20,
            str(count), ha='center', va='bottom', fontsize=12, fontweight='bold'
        )
    axes[1].set_title('流失客户数量', fontsize=13)
    axes[1].set_ylabel('客户数')
    axes[1].set_ylim(0, churn_counts.max() * 1.15)
    axes[1].grid(axis='y', alpha=0.3)

    return _save(fig, '01_churn_distribution.png')


def plot_numerical_features(df: pd.DataFrame):
    """数值特征在流失/留存两组中的分布对比"""
    num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    col_names_cn = {'tenure': '在网时长(月)', 'MonthlyCharges': '月消费(元)', 'TotalCharges': '总消费(元)'}

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle('数值特征在流失/留存客户中的分布对比', fontsize=14, fontweight='bold')

    colors = {'No': '#2ecc71', 'Yes': '#e74c3c'}
    for ax, col in zip(axes, num_cols):
        for churn_val, color in colors.items():
            subset = df[df['Churn'] == churn_val][col].dropna()
            ax.hist(subset, bins=30, alpha=0.6, color=color,
                    label=f"{'留存' if churn_val == 'No' else '流失'}", density=True)
        ax.set_title(col_names_cn.get(col, col), fontsize=12)
        ax.set_xlabel('')
        ax.set_ylabel('密度')
        ax.legend()
        ax.grid(alpha=0.3)

    return _save(fig, '02_numerical_features.png')


def plot_categorical_churn_rate(df: pd.DataFrame):
    """关键类别特征的流失率柱状图"""
    cat_cols = ['Contract', 'InternetService', 'PaymentMethod', 'gender']
    col_names_cn = {
        'Contract'      : '合同类型',
        'InternetService': '网络服务类型',
        'PaymentMethod' : '付款方式',
        'gender'        : '性别',
    }

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('关键类别特征的客户流失率', fontsize=15, fontweight='bold')
    axes = axes.flatten()

    for ax, col in zip(axes, cat_cols):
        churn_rate = df.groupby(col)['Churn'].apply(
            lambda x: (x == 'Yes').mean()
        ).sort_values(ascending=False)

        bars = ax.bar(
            churn_rate.index, churn_rate.values * 100,
            color=plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(churn_rate))),
            edgecolor='white', linewidth=1.2
        )
        for bar, rate in zip(bars, churn_rate.values):
            ax.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f'{rate:.1%}', ha='center', va='bottom', fontsize=9, fontweight='bold'
            )
        ax.set_title(col_names_cn.get(col, col), fontsize=12)
        ax.set_ylabel('流失率 (%)')
        ax.set_ylim(0, churn_rate.max() * 120)
        ax.tick_params(axis='x', rotation=20)
        ax.grid(axis='y', alpha=0.3)

    return _save(fig, '03_categorical_churn_rate.png')


def plot_correlation_heatmap(df: pd.DataFrame):
    """数值特征相关性热力图"""
    try:
        import seaborn as sns
    except ImportError:
        print("[跳过] 相关性热力图需要 seaborn，请 pip install seaborn")
        return

    num_df = df.select_dtypes(include=[np.number])
    if 'SeniorCitizen' not in num_df.columns:
        pass  # 已在原始数据中

    fig, ax = plt.subplots(figsize=(10, 8))
    corr = num_df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(
        corr, mask=mask, annot=True, fmt='.2f',
        cmap='RdBu_r', center=0, linewidths=0.5,
        ax=ax, cbar_kws={'shrink': 0.8}
    )
    ax.set_title('特征相关性矩阵', fontsize=14, fontweight='bold')
    return _save(fig, '04_correlation_heatmap.png')


# ──────────────────────────────────────────────
# 2. 模型性能可视化
# ──────────────────────────────────────────────

def plot_roc_curves(models_dict: dict, X_test, y_test):
    """绘制多模型 ROC 曲线对比"""
    fig, ax = plt.subplots(figsize=(9, 7))

    colors = ['#e74c3c', '#3498db', '#2ecc71', '#9b59b6', '#f39c12']
    for (name, model), color in zip(models_dict.items(), colors):
        y_proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=color, lw=2,
                label=f'{name}  (AUC = {roc_auc:.4f})')

    ax.plot([0, 1], [0, 1], 'k--', lw=1.2, label='随机猜测 (AUC = 0.5000)')
    ax.fill_between([0, 1], [0, 1], alpha=0.05, color='gray')
    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.05])
    ax.set_xlabel('假正率 (FPR)', fontsize=12)
    ax.set_ylabel('真正率 (TPR)', fontsize=12)
    ax.set_title('多模型 ROC 曲线对比', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(alpha=0.3)

    return _save(fig, '05_roc_curves.png')


def plot_confusion_matrix(model, X_test, y_test, model_name: str = 'Best Model'):
    """绘制混淆矩阵"""
    y_pred = model.predict(X_test)
    cm     = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['留存', '流失'])
    disp.plot(ax=ax, colorbar=False, cmap='Blues')
    ax.set_title(f'混淆矩阵 ({model_name})', fontsize=13, fontweight='bold')

    return _save(fig, '06_confusion_matrix.png')


def plot_metrics_comparison(all_metrics: list):
    """多模型评估指标雷达图 + 柱状对比图"""
    metrics_keys = ['accuracy', 'precision', 'recall', 'f1', 'auc']
    labels_cn    = ['准确率', '精确率', '召回率', 'F1', 'AUC']

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('多模型性能对比', fontsize=15, fontweight='bold')

    # ── 柱状对比图 ──
    ax = axes[0]
    x     = np.arange(len(metrics_keys))
    width = 0.15
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#9b59b6', '#f39c12']

    for i, (m_dict, color) in enumerate(zip(all_metrics, colors)):
        vals = [m_dict[k] for k in metrics_keys]
        bars = ax.bar(x + i * width, vals, width, label=m_dict['model'],
                      color=color, alpha=0.85, edgecolor='white')

    ax.set_xticks(x + width * (len(all_metrics) - 1) / 2)
    ax.set_xticklabels(labels_cn, fontsize=11)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel('分数')
    ax.set_title('各指标柱状对比', fontsize=13)
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(axis='y', alpha=0.3)

    # ── AUC 排名条形图 ──
    ax2 = axes[1]
    model_names = [m['model'] for m in all_metrics]
    auc_vals    = [m['auc'] for m in all_metrics]
    sorted_pairs = sorted(zip(auc_vals, model_names), reverse=True)
    auc_sorted, name_sorted = zip(*sorted_pairs)

    bar_colors = ['#e74c3c' if v == max(auc_sorted) else '#3498db' for v in auc_sorted]
    bars = ax2.barh(name_sorted, auc_sorted, color=bar_colors,
                    edgecolor='white', linewidth=1.2)
    for bar, val in zip(bars, auc_sorted):
        ax2.text(val + 0.002, bar.get_y() + bar.get_height() / 2,
                 f'{val:.4f}', va='center', fontsize=10, fontweight='bold')
    ax2.set_xlim(0.5, 1.05)
    ax2.set_xlabel('AUC-ROC')
    ax2.set_title('模型 AUC 排名', fontsize=13)
    ax2.grid(axis='x', alpha=0.3)

    return _save(fig, '07_metrics_comparison.png')


def plot_feature_importance(feature_imp_df: pd.DataFrame, top_n: int = 15):
    """特征重要性水平条形图"""
    if feature_imp_df is None or feature_imp_df.empty:
        return

    top_df = feature_imp_df.head(top_n).sort_values('importance')
    fig, ax = plt.subplots(figsize=(10, 7))

    colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(top_df)))
    bars = ax.barh(top_df['feature'], top_df['importance'],
                   color=colors[::-1], edgecolor='white', linewidth=0.8)
    for bar, val in zip(bars, top_df['importance']):
        ax.text(val + 0.001, bar.get_y() + bar.get_height() / 2,
                f'{val:.4f}', va='center', fontsize=9)

    ax.set_xlabel('特征重要性', fontsize=12)
    ax.set_title(f'Top {top_n} 重要特征（基于最优模型）', fontsize=13, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)

    return _save(fig, '08_feature_importance.png')


# ──────────────────────────────────────────────
# 3. 预警系统可视化
# ──────────────────────────────────────────────

def plot_risk_distribution(report: pd.DataFrame):
    """客户风险等级分布图"""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle('客户流失风险分布', fontsize=15, fontweight='bold')

    risk_counts = report['risk_level'].value_counts()
    level_order = ['HIGH', 'MEDIUM', 'LOW']
    colors_map  = {'HIGH': '#e74c3c', 'MEDIUM': '#f39c12', 'LOW': '#2ecc71'}
    labels_cn   = {'HIGH': '高风险', 'MEDIUM': '中风险', 'LOW': '低风险'}

    counts = [risk_counts.get(l, 0) for l in level_order]
    colors = [colors_map[l] for l in level_order]
    labels = [labels_cn[l] for l in level_order]

    # 饼图
    wedges, texts, autotexts = axes[0].pie(
        counts, labels=labels, colors=colors,
        autopct='%1.1f%%', startangle=90,
        textprops={'fontsize': 12},
        wedgeprops={'linewidth': 2, 'edgecolor': 'white'}
    )
    axes[0].set_title('风险等级比例', fontsize=13)

    # 流失概率分布直方图（按风险等级染色）
    for level, color, label in zip(level_order, colors, labels):
        subset = report[report['risk_level'] == level]['churn_proba']
        if len(subset) > 0:
            axes[1].hist(subset, bins=20, alpha=0.7, color=color,
                         label=f'{label} (n={len(subset)})', density=False)

    axes[1].axvline(0.40, color='#f39c12', linestyle='--', linewidth=1.5, label='中/低 阈值(0.40)')
    axes[1].axvline(0.70, color='#e74c3c', linestyle='--', linewidth=1.5, label='高/中 阈值(0.70)')
    axes[1].set_xlabel('流失概率', fontsize=12)
    axes[1].set_ylabel('客户数')
    axes[1].set_title('流失概率分布（按风险等级）', fontsize=13)
    axes[1].legend(fontsize=9)
    axes[1].grid(alpha=0.3)

    return _save(fig, '09_risk_distribution.png')


def plot_top_risk_customers(report: pd.DataFrame, top_n: int = 20):
    """高风险客户流失概率排行榜"""
    top_df = report.head(top_n).copy()
    top_df = top_df.sort_values('churn_proba')

    fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.35)))
    colors = ['#e74c3c' if r == 'HIGH' else '#f39c12' if r == 'MEDIUM' else '#2ecc71'
              for r in top_df['risk_level']]

    bars = ax.barh(top_df['customerID'], top_df['churn_proba'] * 100,
                   color=colors, edgecolor='white', linewidth=0.8)
    for bar, val in zip(bars, top_df['churn_proba']):
        ax.text(val * 100 + 0.5, bar.get_y() + bar.get_height() / 2,
                f'{val:.1%}', va='center', fontsize=9, fontweight='bold')

    ax.axvline(70, color='#e74c3c', linestyle='--', linewidth=1.5, alpha=0.7)
    ax.axvline(40, color='#f39c12', linestyle='--', linewidth=1.5, alpha=0.7)
    ax.set_xlabel('流失概率 (%)')
    ax.set_title(f'Top {top_n} 高风险客户预警列表', fontsize=13, fontweight='bold')
    ax.set_xlim(0, 110)
    ax.grid(axis='x', alpha=0.3)

    # 图例
    patches = [
        mpatches.Patch(color='#e74c3c', label='高风险'),
        mpatches.Patch(color='#f39c12', label='中风险'),
        mpatches.Patch(color='#2ecc71', label='低风险'),
    ]
    ax.legend(handles=patches, loc='lower right', fontsize=9)

    return _save(fig, '10_top_risk_customers.png')
