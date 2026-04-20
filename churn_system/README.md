# 基于Python的客户流失分析与预警系统

> 毕业设计项目 · 大四 · 计算机科学与技术

---

## 📋 项目简介

本项目实现了一套完整的**客户流失分析与预警系统**，面向电信、金融、零售等行业，通过机器学习手段对客户流失风险进行量化评估，并输出分级预警与挽留策略建议。

**核心功能：**
- 📊 数据自动生成/加载与清洗
- 🔍 探索性数据分析（EDA）与可视化
- 🤖 多模型训练（逻辑回归 / 随机森林 / XGBoost / LightGBM）
- 📈 模型性能对比（ROC、混淆矩阵、F1 等）
- 🚨 三级流失风险预警（高/中/低）
- 💡 个性化客户挽留策略推荐
- 📄 自动导出 CSV 报告与 10 张可视化图表

---

## 📁 项目结构

```
churn_system/
│
├── main.py              # ★ 主程序入口，一键运行完整流程
├── data_loader.py       # 数据生成、清洗、特征编码与划分
├── model_trainer.py     # 多模型训练、交叉验证、调参、保存
├── early_warning.py     # 预警评分、风险分级、策略推荐
├── visualizer.py        # EDA + 模型性能 + 预警 全套可视化
│
├── data/
│   └── mock_customer_data.csv     # 自动生成的模拟数据集
│
├── models/
│   ├── LogisticRegression.pkl     # 逻辑回归模型
│   ├── RandomForest.pkl           # 随机森林模型
│   ├── GradientBoosting.pkl       # 梯度提升模型
│   ├── XGBoost.pkl                # XGBoost 模型（可选）
│   ├── LightGBM.pkl               # LightGBM 模型（可选）
│   ├── scaler.pkl                 # 数值标准化器
│   ├── encoders.pkl               # 类别编码器
│   └── feature_names.pkl          # 特征名称列表
│
└── output/
    ├── 01_churn_distribution.png  # 流失分布饼图+柱图
    ├── 02_numerical_features.png  # 数值特征对比直方图
    ├── 03_categorical_churn_rate.png  # 类别特征流失率
    ├── 04_correlation_heatmap.png # 相关性热力图
    ├── 05_roc_curves.png          # 多模型 ROC 曲线
    ├── 06_confusion_matrix.png    # 最优模型混淆矩阵
    ├── 07_metrics_comparison.png  # 多模型指标对比
    ├── 08_feature_importance.png  # 特征重要性排名
    ├── 09_risk_distribution.png   # 风险等级分布
    ├── 10_top_risk_customers.png  # 高风险客户排行
    ├── churn_warning_report_*.csv # 完整预警报告
    ├── cv_results.csv             # 交叉验证结果
    └── feature_importance.csv     # 特征重要性表格
```

---

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install numpy pandas scikit-learn matplotlib seaborn
# 可选（增加更多模型）
pip install xgboost lightgbm
```

### 2. 运行系统

```bash
cd churn_system
python main.py
```

运行完毕后，所有图表和报告自动保存到 `output/` 目录。

---

## 🧠 技术方案

### 数据集

| 字段 | 含义 |
|------|------|
| tenure | 客户在网时长（月） |
| MonthlyCharges | 月消费金额 |
| Contract | 合同类型（月付/年付/两年） |
| InternetService | 网络服务类型 |
| Churn | **目标变量**（Yes=流失 / No=留存） |

> 默认使用系统自动生成的 5000 条模拟数据，也可替换为 Kaggle Telco Customer Churn 数据集。

### 特征工程

- **二值编码**：Yes/No → 1/0
- **标签编码**：多类别特征 → LabelEncoder
- **标准化**：数值型特征 → StandardScaler（零均值单位方差）

### 模型列表

| 模型 | 库 | 特点 |
|------|----|------|
| 逻辑回归 | sklearn | 可解释性强，基线模型 |
| 随机森林 | sklearn | 鲁棒性好，支持特征重要性 |
| 梯度提升树 | sklearn | 精度高，抗过拟合 |
| XGBoost | xgboost | 工业级 GBDT，速度快 |
| LightGBM | lightgbm | 大数据场景首选 |

### 预警规则

| 等级 | 流失概率 | 策略 |
|------|----------|------|
| 🔴 高风险 | ≥ 70% | 立即回访 + 专属优惠 |
| 🟡 中风险 | 40%~70% | 定向营销 + 合同转换奖励 |
| 🟢 低风险 | < 40% | 常规运营 + 满意度调研 |

---

## 📊 系统流程图

```
原始数据
   ↓
数据清洗 & 预处理
   ↓
EDA 可视化 (4 张图)
   ↓
特征编码 & 标准化 & 数据集划分
   ↓
多模型训练 + 5折交叉验证
   ↓
模型评估 & 可视化 (4 张图)
   ↓
生成预警报告 + 风险评级
   ↓
预警可视化 (2 张图) + CSV 导出
   ↓
单客户实时预警演示
```

---

## 📌 注意事项

1. 若未安装 `seaborn`，相关性热力图会自动跳过
2. 若未安装 `xgboost` / `lightgbm`，系统仍可正常运行（只是少对应模型）
3. Windows 下如遇中文乱码，确保 `Microsoft YaHei` 字体已安装
4. 如需使用真实数据，将 `main.py` 中 `generate_mock_data` 替换为 `pd.read_csv('your_data.csv')`

---

## 🎓 学术参考

- Breiman, L. (2001). Random Forests. *Machine Learning*, 45, 5–32.
- Chen, T., & Guestrin, C. (2016). XGBoost. *KDD '16*.
- Ke, G. et al. (2017). LightGBM. *NeurIPS*.
- Kaggle Telco Customer Churn Dataset: https://www.kaggle.com/blastchar/telco-customer-churn

---

*本项目用于毕业设计，代码结构完整，注释详尽，欢迎参考与扩展。*
