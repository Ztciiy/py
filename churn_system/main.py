"""
=============================================================
主程序: main.py
功能:  基于 Python 的客户流失分析与预警系统 —— 完整运行入口
作者:  大四毕业设计项目
时间:  2026 年

系统流程:
  1. 生成/加载数据集
  2. 数据清洗与特征预处理
  3. EDA 可视化
  4. 多模型训练、交叉验证、调参
  5. 模型评估与可视化
  6. 构建预警报告
  7. 导出报告与图表
=============================================================
"""

import os
import sys
import time
import pandas as pd

# ── 将当前目录加入路径（确保子模块可导入）──
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_loader   import generate_mock_data, clean_data, preprocess_features, split_dataset, load_data_auto
from model_trainer import (train_all_models, get_model_zoo, evaluate_model,
                            get_feature_importance, load_model, save_model)
from early_warning import (build_warning_report, summarize_warning,
                            export_report, export_warning_list,
                            single_customer_warning, batch_ai_retention,
                            analyze_churn_reasons)
from visualizer    import (plot_churn_distribution, plot_numerical_features,
                            plot_categorical_churn_rate, plot_correlation_heatmap,
                            plot_roc_curves, plot_confusion_matrix,
                            plot_metrics_comparison, plot_feature_importance,
                            plot_risk_distribution, plot_top_risk_customers)


# ──────────────────────────────────────────────
# 系统参数配置
# ──────────────────────────────────────────────

CONFIG = {
    'n_samples'     : 5000,          # 模拟数据量（仅在不使用真实CSV时生效）
    'test_size'     : 0.20,          # 测试集比例
    'val_size'      : 0.10,          # 验证集比例
    'random_state'  : 42,            # 随机种子
    'model_dir'     : 'models',      # 模型保存目录
    'output_dir'    : 'output',      # 报告 & 图表输出目录
    'data_save_path': 'data/mock_customer_data.csv',  # 数据保存路径

    # ── 真实数据 CSV 配置 ──────────────────────────────────────────
    # 用法一：指定具体文件列表（支持多个）
    #   'csv_paths': ['data/customers_2024.csv', 'data/customers_2025.csv'],
    #
    # 用法二：指定文件夹，自动扫描所有 .csv
    #   'csv_paths': 'data/',
    #
    # 用法三：设为 None，使用系统内置模拟数据（默认）
    # 'csv_paths'     : None,  # 设为None则使用模拟数据

    # ★★★ 重要 ★★★ 使用Kaggle真实格式数据！
    'csv_paths'     : ['data/telco_churn.csv'],  # 7043条Kaggle格式数据，流失率27.4%
    # ─────────────────────────────────────────────────────────────

    # ── 预警名单导出配置 ──────────────────────────────────────────
    'export_format' : 'csv',   # 'csv' 或 'excel'（需安装 openpyxl）
    'export_top_n'  : None,    # None=导出全部；50=只导出前50名高风险
    # ─────────────────────────────────────────────────────────────

    # ── AI 接口配置（可选，不填则使用规则兜底方案）────────────────
    # 填入你的 API Key 即可启用 AI 个性化挽留方案功能
    # 兼容 OpenAI 格式，支持以下服务：
    #   - SiliconFlow（推荐，注册送100元额度，国产模型中文好）
    #   - OpenAI 官方 / 腾讯混元 / 阿里通义 / 讯飞星火 等
    #
    # SiliconFlow 配置示例（推荐）：
    #   'ai_api_key' : 'sk-xxxxxxxxxxxxxxxxxxxx',
    #   'ai_api_base': 'https://api.siliconflow.cn/v1',
    #   'ai_model'   : 'Qwen/Qwen2.5-7B-Instruct',
    #
    # OpenAI 官方示例:
    #   'ai_api_key' : 'sk-xxxxxxxxxxxxxxxxxxxx',
    #   'ai_api_base': 'https://api.openai.com/v1',
    #   'ai_model'   : 'gpt-3.5-turbo',
    #
    # 腾讯混元示例:
    #   'ai_api_key' : 'your-key',
    #   'ai_api_base': 'https://api.hunyuan.cloud.tencent.com/v1',
    #   'ai_model'   : 'hunyuan-pro',
    #
    'ai_api_key'    : 'sk-pferjuznjmupexuzqtirjzwxqppptanidhyccijubfdjqubc',    # 填入 API Key 启用 AI 功能
    'ai_api_base'   : 'https://api.siliconflow.cn/v1',  # SiliconFlow（推荐）
    'ai_model'      : 'Qwen/Qwen2.5-7B-Instruct',  # SiliconFlow实测可用
    'ai_top_n'      : 10,       # 为前10名高风险客户生成 AI 方案
    # ─────────────────────────────────────────────────────────────
}


# ──────────────────────────────────────────────
# 打印分隔线工具
# ──────────────────────────────────────────────

def section(title: str):
    print(f"\n{'═'*60}")
    print(f"  STEP: {title}")
    print(f"{'═'*60}")


# ──────────────────────────────────────────────
# 主流程
# ──────────────────────────────────────────────

def main():
    start_time = time.time()
    print("=" * 60)
    print("  [启动] 客户流失分析与预警系统")
    print("=" * 60)

    # ─────────────────────────────────────────
    # STEP 1: 数据加载
    # ─────────────────────────────────────────
    section("数据加载")
    raw_df = load_data_auto(
        csv_paths=CONFIG.get('csv_paths'),
        n_samples=CONFIG['n_samples'],
        random_state=CONFIG['random_state']
    )

    # 删除辅助来源列（不参与模型训练）
    if '_source' in raw_df.columns:
        raw_df.drop('_source', axis=1, inplace=True)

    # 保存合并后的原始数据
    os.makedirs('data', exist_ok=True)
    raw_df.to_csv(CONFIG['data_save_path'], index=False, encoding='utf-8-sig')
    print(f"[数据] 原始数据已保存: {CONFIG['data_save_path']}")

    # ─────────────────────────────────────────
    # STEP 2: 数据清洗
    # ─────────────────────────────────────────
    section("数据清洗")
    clean_df = clean_data(raw_df)
    print(f"[清洗后] 数据集大小: {clean_df.shape}")

    # ─────────────────────────────────────────
    # STEP 3: EDA 可视化
    # ─────────────────────────────────────────
    section("探索性数据分析 (EDA)")
    os.makedirs(CONFIG['output_dir'], exist_ok=True)

    print("[可视化] 绘制流失分布图...")
    plot_churn_distribution(clean_df)

    print("[可视化] 绘制数值特征分布图...")
    plot_numerical_features(clean_df)

    print("[可视化] 绘制类别特征流失率图...")
    plot_categorical_churn_rate(clean_df)

    print("[可视化] 绘制相关性热力图...")
    plot_correlation_heatmap(clean_df)

    # ─────────────────────────────────────────
    # STEP 4: 特征预处理 & 数据集划分
    # ─────────────────────────────────────────
    section("特征预处理与数据集划分")
    X, y, feature_names, scaler, encoders = preprocess_features(clean_df)
    X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(
        X, y,
        test_size=CONFIG['test_size'],
        val_size=CONFIG['val_size'],
        random_state=CONFIG['random_state']
    )

    # ─────────────────────────────────────────
    # STEP 5: 模型训练 & 评估
    # ─────────────────────────────────────────
    section("多模型训练与评估")
    best_model, all_metrics, feature_imp_df, cv_results = train_all_models(
        X_train, y_train, X_test, y_test,
        feature_names=feature_names,
        model_save_dir=CONFIG['model_dir']
    )

    # 保存 scaler 和 encoders（供后续预测使用）
    import pickle
    with open(os.path.join(CONFIG['model_dir'], 'scaler.pkl'), 'wb') as f:
        pickle.dump(scaler, f)
    with open(os.path.join(CONFIG['model_dir'], 'encoders.pkl'), 'wb') as f:
        pickle.dump(encoders, f)
    with open(os.path.join(CONFIG['model_dir'], 'feature_names.pkl'), 'wb') as f:
        pickle.dump(feature_names, f)
    print(f"[保存] scaler / encoders / feature_names 已保存至 {CONFIG['model_dir']}/")

    # ─────────────────────────────────────────
    # STEP 6: 模型性能可视化
    # ─────────────────────────────────────────
    section("模型性能可视化")
    # 加载所有训练好的模型（用于 ROC 多曲线绘制）
    trained_models = {}
    for m in all_metrics:
        model_path = os.path.join(CONFIG['model_dir'], f"{m['model']}.pkl")
        if os.path.exists(model_path):
            trained_models[m['model']] = load_model(model_path)

    print("[可视化] 绘制多模型 ROC 曲线...")
    plot_roc_curves(trained_models, X_test, y_test)

    best_model_name = max(all_metrics, key=lambda x: x['auc'])['model']
    print(f"[可视化] 绘制最优模型 ({best_model_name}) 混淆矩阵...")
    plot_confusion_matrix(best_model, X_test, y_test, model_name=best_model_name)

    print("[可视化] 绘制多模型指标对比图...")
    plot_metrics_comparison(all_metrics)

    print("[可视化] 绘制特征重要性图...")
    plot_feature_importance(feature_imp_df)

    # ─────────────────────────────────────────
    # STEP 7: 构建预警报告
    # ─────────────────────────────────────────
    section("客户流失预警报告生成")

    # 使用全量测试集生成预警报告，附上原始数据字段
    # 恢复测试集对应的原始 DataFrame（用于导出可读字段）
    from sklearn.model_selection import train_test_split as _tts
    _, clean_df_test = _tts(
        clean_df, test_size=CONFIG['test_size'],
        random_state=CONFIG['random_state']
    )
    customer_ids = (
        clean_df_test['customerID'].tolist()
        if 'customerID' in clean_df_test.columns
        else [f'CUST-TEST-{str(i).zfill(5)}' for i in range(len(X_test))]
    )
    warning_report = build_warning_report(
        customer_ids, X_test, best_model,
        feature_names=feature_names,
        raw_df=clean_df_test.reset_index(drop=True),
    )

    # 汇总统计
    summary = summarize_warning(warning_report)

    # 预警可视化
    print("[可视化] 绘制风险等级分布图...")
    plot_risk_distribution(warning_report)

    print("[可视化] 绘制 Top 高风险客户列表...")
    plot_top_risk_customers(warning_report, top_n=20)

    # ─────────────────────────────────────────
    # STEP 8: 导出报告（多种格式）
    # ─────────────────────────────────────────
    section("导出报告")

    # 8.1 导出完整预警名单（全部风险等级，按概率排序）
    report_path = export_warning_list(
        warning_report,
        output_dir=CONFIG['output_dir'],
        fmt=CONFIG.get('export_format', 'csv'),
        top_n=CONFIG.get('export_top_n'),
    )

    # 8.2 单独导出高风险客户名单
    high_path = export_warning_list(
        warning_report,
        output_dir=CONFIG['output_dir'],
        risk_filter='HIGH',
        fmt=CONFIG.get('export_format', 'csv'),
    )
    print(f"[导出] 高风险客户名单: {high_path}")

    # 8.3 保存交叉验证结果
    cv_save = os.path.join(CONFIG['output_dir'], 'cv_results.csv')
    cv_results.to_csv(cv_save, index=False, encoding='utf-8-sig')
    print(f"[导出] 交叉验证结果已保存: {cv_save}")

    # 8.4 保存特征重要性
    fi_save = os.path.join(CONFIG['output_dir'], 'feature_importance.csv')
    feature_imp_df.to_csv(fi_save, index=False, encoding='utf-8-sig')
    print(f"[导出] 特征重要性已保存: {fi_save}")

    # ─────────────────────────────────────────
    # STEP 9: AI 批量挽留方案（高风险客户）
    # ─────────────────────────────────────────
    section("AI 个性化挽留方案生成（高风险客户）")

    # 重建 customerID → X 行映射（clean_df_test 与 X_test 同序）
    ai_result = batch_ai_retention(
        warning_report,
        X_test,
        best_model,
        feature_names,
        raw_df=clean_df_test.reset_index(drop=True),
        top_n_customers=CONFIG.get('ai_top_n', 5),
        risk_level='HIGH',
        api_key=CONFIG.get('ai_api_key'),
        api_base=CONFIG.get('ai_api_base'),
        model_name=CONFIG.get('ai_model', 'Qwen/Qwen2.5-7B-Instruct'),
        output_dir=CONFIG['output_dir'],
    )

    # ─────────────────────────────────────────
    # STEP 10: 单客户实时预警演示
    # ─────────────────────────────────────────
    section("单客户实时预警演示（含原因分析）")

    demo_customer = {
        'customerID'      : 'DEMO-001',
        'gender'          : 'Male',
        'SeniorCitizen'   : 0,
        'Partner'         : 'No',
        'Dependents'      : 'No',
        'tenure'          : 3,
        'PhoneService'    : 'Yes',
        'MultipleLines'   : 'No',
        'InternetService' : 'Fiber optic',
        'OnlineSecurity'  : 'No',
        'OnlineBackup'    : 'No',
        'DeviceProtection': 'No',
        'TechSupport'     : 'No',
        'StreamingTV'     : 'Yes',
        'StreamingMovies' : 'Yes',
        'Contract'        : 'Month-to-month',
        'PaperlessBilling': 'Yes',
        'PaymentMethod'   : 'Electronic check',
        'MonthlyCharges'  : 95.5,
        'TotalCharges'    : 286.5,
    }

    print("\n[演示] 客户信息:")
    for k, v in demo_customer.items():
        print(f"    {k:25s}: {v}")

    result = single_customer_warning(
        demo_customer, best_model, scaler, encoders, feature_names,
        api_key=CONFIG.get('ai_api_key'),
        api_base=CONFIG.get('ai_api_base'),
        model_name=CONFIG.get('ai_model', 'Qwen/Qwen2.5-7B-Instruct'),
    )

    # ─────────────────────────────────────────
    # 结束统计
    # ─────────────────────────────────────────
    elapsed = time.time() - start_time
    # ─────────────────────────────────────────
    # STEP 11: 生成仪表盘数据文件
    # ─────────────────────────────────────────
    try:
        from generate_dashboard_js import generate_dashboard_js
        generate_dashboard_js()
        print(f"[仪表盘] 数据文件已更新")
    except Exception as e:
        print(f"[仪表盘] 数据文件生成失败: {e}")

    print(f"\n{'='*60}")
    print(f"  [完成] 系统运行完成！总耗时: {elapsed:.1f} 秒")
    print(f"  [输出] 输出目录: {os.path.abspath(CONFIG['output_dir'])}")
    print(f"  [图表] 图表数量: 10 张")
    print(f"  [报告] 完整预警名单: {report_path}")
    print(f"  [报告] 高风险客户名单: {high_path}")
    print(f"  [AI]   高风险客户 AI 挽留方案: output/ai_retention_plans_high_*.csv")
    print(f"  [仪表盘] 打开 dashboard_final.html 查看可视化结果")
    print(f"{'='*60}\n")


# ──────────────────────────────────────────────
# 程序入口
# ──────────────────────────────────────────────

if __name__ == '__main__':
    main()
