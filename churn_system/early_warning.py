"""
=============================================================
模块: early_warning.py
功能: 客户流失预警系统核心模块
作者: 大四毕业设计项目
说明: 对客户进行流失概率评分、风险分级、原因归因分析，
      支持导出预警名单和接入 AI 接口生成个性化挽留方案
=============================================================
"""

import os
import json
import pickle
import pandas as pd
import numpy as np
from datetime import datetime


# ──────────────────────────────────────────────
# 1. 综合风险等级划分规则
# 综合风险评分 = 流失概率 × 月消费权重 × 流失原因权重
# ──────────────────────────────────────────────

# 月消费权重因子
MONTHLY_CHARGE_WEIGHTS = {
    'HIGH'  : 1.3,    # >100元：流失影响最大
    'MEDIUM': 1.2,    # 70-100元
    'LOW'   : 1.1,    # 50-70元
    'MIN'   : 1.0,    # <50元
}

# 流失原因权重加成
REASON_WEIGHTS = {
    '合同到期未续约'      : 0.15,
    '年付合同即将到期'   : 0.12,
    '月租费用过高'       : 0.10,
    '月租费用偏高'       : 0.10,
    '流失概率极高'       : 0.10,
    '需立即挽留'         : 0.10,
    '网络质量不稳定'     : 0.08,
    '光纤用户对网络质量要求高': 0.05,
    '缺少技术支持服务'  : 0.05,
    '无网络服务'         : 0.05,
    '增值业务少'         : 0.05,
    '在网时间较短'       : 0.05,
    '电子支票支付'       : 0.05,
    '账单拖欠'           : 0.05,
    '邮寄支票支付'       : 0.03,
    '月付合同'           : 0.05,
}

# 综合风险评分阈值
RISK_THRESHOLDS = {
    'HIGH'  : 0.55,   # 高风险：综合评分 >= 0.55
    'MEDIUM': 0.30,   # 中风险：0.30 <= 综合评分 < 0.55
    'LOW'   : 0.00,   # 低风险：综合评分 < 0.30
}

RISK_LABELS_CN = {
    'HIGH'  : '[高风险]',
    'MEDIUM': '[中风险]',
    'LOW'   : '[低风险]',
}

# 各风险等级基础挽留策略（备用，AI 未启用时使用）
RETENTION_STRATEGIES = {
    'HIGH': [
        '① 立即分配专属客服进行一对一回访',
        '② 提供个性化优惠套餐（折扣率 20%-30%）',
        '③ 优先升级服务（免费提速 / 免费附加服务 3 个月）',
        '④ 赠送积分或消费奖励',
        '⑤ 安排高管关怀电话，倾听诉求',
    ],
    'MEDIUM': [
        '① 发送定向营销短信或邮件，推送优惠活动',
        '② 提供年度合同转换奖励（折扣 10%-15%）',
        '③ 定期推送使用报告，增加用户黏性',
        '④ 邀请参与忠诚度积分计划',
    ],
    'LOW': [
        '① 保持常规运营通知与关怀',
        '② 定期满意度调研，收集反馈',
        '③ 推送新产品 / 功能介绍',
    ],
}

# 特征中文名映射（用于展示）
FEATURE_NAME_CN = {
    'tenure'          : '在网时长(月)',
    'MonthlyCharges'  : '月消费金额',
    'TotalCharges'    : '累计消费金额',
    'Contract'        : '合同类型',
    'InternetService' : '网络服务类型',
    'PaymentMethod'   : '支付方式',
    'OnlineSecurity'  : '在线安全服务',
    'TechSupport'     : '技术支持服务',
    'OnlineBackup'    : '在线备份服务',
    'DeviceProtection': '设备保护服务',
    'PaperlessBilling': '电子账单',
    'MultipleLines'   : '多线服务',
    'StreamingTV'     : '流媒体电视',
    'StreamingMovies' : '流媒体电影',
    'Partner'         : '是否有伴侣',
    'Dependents'      : '是否有家属',
    'SeniorCitizen'   : '是否老年用户',
    'gender'          : '性别',
    'PhoneService'    : '电话服务',
}


# ──────────────────────────────────────────────
# 2. 流失概率预测与风险评级
# ──────────────────────────────────────────────

def predict_churn_proba(model, X: np.ndarray) -> np.ndarray:
    """使用训练好的模型预测流失概率"""
    return model.predict_proba(X)[:, 1]


def get_monthly_weight(monthly_charge: float) -> float:
    """获取月消费权重因子"""
    if monthly_charge > 100:
        return MONTHLY_CHARGE_WEIGHTS['HIGH']
    elif monthly_charge > 70:
        return MONTHLY_CHARGE_WEIGHTS['MEDIUM']
    elif monthly_charge > 50:
        return MONTHLY_CHARGE_WEIGHTS['LOW']
    else:
        return MONTHLY_CHARGE_WEIGHTS['MIN']


def get_reason_weight(churn_reason: str) -> float:
    """获取流失原因权重加成"""
    weight = 1.0
    reason = str(churn_reason)
    for key, val in REASON_WEIGHTS.items():
        if key in reason:
            weight += val
    return weight


def calculate_comprehensive_score(churn_proba: float, monthly_charge: float, churn_reason: str) -> float:
    """
    计算综合风险评分
    综合风险评分 = 流失概率 × 月消费权重 × 流失原因权重
    """
    monthly_weight = get_monthly_weight(monthly_charge)
    reason_weight = get_reason_weight(churn_reason)
    
    comprehensive_score = churn_proba * monthly_weight * reason_weight
    # 限制在0-1之间
    comprehensive_score = min(comprehensive_score, 1.0)
    
    return comprehensive_score


def assign_risk_level(churn_proba: float, monthly_charge: float = None, churn_reason: str = None) -> tuple:
    """
    根据综合评分判断风险等级
    
    参数:
        churn_proba: 流失概率 (0-1)
        monthly_charge: 月消费金额 (可选，不传则只用流失概率)
        churn_reason: 流失原因分析 (可选)
    
    返回:
        (风险等级代码, 综合评分, 月消费权重, 原因权重)
        例如: ('HIGH', 0.85, 1.3, 1.25)
    """
    if monthly_charge is not None and churn_reason is not None:
        # 综合评分模式
        comprehensive_score = calculate_comprehensive_score(churn_proba, monthly_charge, churn_reason)
        monthly_weight = get_monthly_weight(monthly_charge)
        reason_weight = get_reason_weight(churn_reason)
    else:
        # 兼容旧模式：仅基于流失概率
        comprehensive_score = churn_proba
        monthly_weight = 1.0
        reason_weight = 1.0
    
    if comprehensive_score >= RISK_THRESHOLDS['HIGH']:
        return 'HIGH', comprehensive_score, monthly_weight, reason_weight
    elif comprehensive_score >= RISK_THRESHOLDS['MEDIUM']:
        return 'MEDIUM', comprehensive_score, monthly_weight, reason_weight
    else:
        return 'LOW', comprehensive_score, monthly_weight, reason_weight


# ──────────────────────────────────────────────
# 3. 单客户流失原因归因分析
# ──────────────────────────────────────────────

def analyze_churn_reasons(customer_row: np.ndarray,
                           model,
                           feature_names: list,
                           raw_customer_data: dict = None,
                           top_n: int = 5) -> dict:
    """
    对单个客户进行流失原因归因分析。

    方法：
      - 树模型：使用全局 feature_importances_ 结合客户特征值定向分析
      - 通用：基于排列重要性思想逐特征扰动，计算各特征对预测的贡献度

    参数:
        customer_row     : 单行特征矩阵 (1, n_features)
        model            : 已训练模型
        feature_names    : 特征名称列表
        raw_customer_data: 客户原始特征字典（用于生成可读描述）
        top_n            : 返回主要原因数量

    返回:
        dict: {
            'top_reasons': [(特征名, 重要度, 原始值, 描述), ...],
            'analysis_text': 文字总结
        }
    """
    X = customer_row.reshape(1, -1) if customer_row.ndim == 1 else customer_row
    base_proba = model.predict_proba(X)[0, 1]

    contributions = []

    # ── 方法一：基于树模型全局特征重要性 + 高危特征值判断 ──
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        for i, (name, imp) in enumerate(zip(feature_names, importances)):
            val = float(X[0, i])
            # 对重要性做加权（值越大/越异常，风险越高）
            # 用客户在该特征上的标准化值（已经过StandardScaler）
            # 正值表示偏高（对流失方向贡献），取绝对值的重要度
            contribution = imp * (1 + abs(val))
            contributions.append((name, imp, val, contribution))
    else:
        # ── 方法二：逐特征扰动法（通用） ──
        for i, name in enumerate(feature_names):
            X_perturb = X.copy()
            X_perturb[0, i] = 0  # 归零扰动
            perturb_proba = model.predict_proba(X_perturb)[0, 1]
            contribution = abs(base_proba - perturb_proba)
            contributions.append((name, None, float(X[0, i]), contribution))

    # 按贡献度降序排列，取 top_n
    contributions.sort(key=lambda x: x[3], reverse=True)
    top_reasons = contributions[:top_n]

    # 生成可读描述
    reasons_readable = []
    for name, imp, val, contrib in top_reasons:
        cn_name = FEATURE_NAME_CN.get(name, name)
        # 获取原始值（未标准化的）
        raw_val = None
        if raw_customer_data and name in raw_customer_data:
            raw_val = raw_customer_data[name]

        if raw_val is not None:
            desc = f"{cn_name}：{raw_val}"
        else:
            # 根据标准化值判断方向
            direction = "偏高" if val > 0 else ("偏低" if val < -0.5 else "正常")
            desc = f"{cn_name}（{direction}）"

        reasons_readable.append({
            'feature'     : name,
            'feature_cn'  : cn_name,
            'raw_value'   : raw_val,
            'contribution': round(contrib, 4),
            'description' : desc,
        })

    # 生成文字总结
    reason_strs = [r['description'] for r in reasons_readable[:3]]
    analysis_text = (
        f"该客户流失概率为 {base_proba:.1%}，"
        f"主要风险因素为：{'、'.join(reason_strs)}。"
    )

    return {
        'top_reasons'  : reasons_readable,
        'base_proba'   : round(base_proba, 4),
        'analysis_text': analysis_text,
    }


# ──────────────────────────────────────────────
# 4. 构建批量预警报告
# ──────────────────────────────────────────────

def build_warning_report(customer_ids: list,
                          X: np.ndarray,
                          model,
                          feature_names: list = None,
                          raw_df: pd.DataFrame = None,
                          top_n: int = 5) -> pd.DataFrame:
    """
    对一批客户生成完整预警报告（按风险从高到低排序）

    参数:
        customer_ids: 客户 ID 列表
        X           : 特征矩阵
        model       : 已训练模型
        feature_names: 特征名列表
        raw_df      : 原始数据 DataFrame（用于附加原始字段到报告）
        top_n       : 每位客户显示贡献最大的前 N 个特征

    返回:
        DataFrame: 预警报告（已按 churn_proba 降序排列）
    """
    probas      = predict_churn_proba(model, X)
    # assign_risk_level 返回 (level_code, score, ...) 元组，取第一个元素
    risk_level_codes = [assign_risk_level(p)[0] for p in probas]

    report = pd.DataFrame({
        'customerID'   : customer_ids,
        'churn_proba'  : np.round(probas, 4),
        'risk_level'   : risk_level_codes,
        'risk_label_cn': [RISK_LABELS_CN[r] for r in risk_level_codes],
        'strategy'     : ['\n'.join(RETENTION_STRATEGIES[r][:2]) for r in risk_level_codes],
        'report_time'  : datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    })

    # ── 附加：Top 风险因素（树模型专用） ──
    if feature_names and hasattr(model, 'feature_importances_'):
        fi = model.feature_importances_
        top_indices   = np.argsort(fi)[::-1][:top_n]
        top_feat_str  = ' | '.join([
            FEATURE_NAME_CN.get(feature_names[i], feature_names[i])
            for i in top_indices
        ])
        report['top_risk_factors'] = top_feat_str

    # ── 附加原始数据关键字段（如月消费、合同类型等） ──
    if raw_df is not None and len(raw_df) == len(report):
        raw_df_reset = raw_df.reset_index(drop=True)
        key_cols = [
            'tenure', 'MonthlyCharges', 'TotalCharges',
            'Contract', 'InternetService', 'PaymentMethod',
            # 新增真实信息字段
            'customerName', 'gender', 'phone', 'email',
            'province', 'city', 'district', 'address', 'churnReason'
        ]
        for col in key_cols:
            if col in raw_df_reset.columns:
                report[col] = raw_df_reset[col].values

    # ── 为缺失流失原因的客户生成基于特征的归因分析 ──
    # 只有实际流失(Churn=='Yes')的客户才有原始原因，高风险预测客户需要基于特征生成
    for idx, row in report.iterrows():
        if pd.isna(row.get('churnReason')) or str(row.get('churnReason')).strip() == '':
            # 基于客户特征生成流失原因归因
            reasons = []
            proba = row.get('churn_proba', 0)

            # 月消费高
            monthly = row.get('MonthlyCharges', 0)
            if monthly > 80:
                reasons.append(f'月租费用偏高(¥{monthly:.0f}/月)')
            elif monthly > 60:
                reasons.append(f'月租费用中等偏高(¥{monthly:.0f}/月)')

            # 在网时间短
            tenure = row.get('tenure', 0)
            if tenure < 12:
                reasons.append(f'在网时间较短({tenure}个月)，客户黏性不足')
            elif tenure < 24:
                reasons.append(f'在网时间({tenure}个月)，仍处于流失高风险期')

            # 合同类型
            contract = row.get('Contract', '')
            if contract == 'Month-to-month':
                reasons.append('月付合同，转换成本低，易流失')
            elif contract == 'One year':
                reasons.append('年付合同即将到期，需重点关注')

            # 网络服务
            internet = row.get('InternetService', '')
            if internet == 'Fiber optic':
                reasons.append('光纤用户对网络质量要求高')
            elif internet == 'No':
                reasons.append('无网络服务，增值业务少')

            # 支付方式风险
            payment = row.get('PaymentMethod', '')
            if payment == 'Electronic check':
                reasons.append('电子支票支付，账单拖欠风险较高')
            elif payment == 'Mailed check':
                reasons.append('邮寄支票支付，便捷性差')

            # 其他附加服务缺失
            if proba > 0.7:
                reasons.append('流失概率极高，需立即挽留')

            # 去重并限制数量
            reasons = list(dict.fromkeys(reasons))[:4]

            if reasons:
                report.at[idx, 'churnReason'] = '；'.join(reasons)
            else:
                report.at[idx, 'churnReason'] = '综合风险因素，需进一步分析'

    # 按流失概率降序排列
    report = report.sort_values('churn_proba', ascending=False).reset_index(drop=True)
    report.insert(0, 'rank', range(1, len(report) + 1))

    return report


# ──────────────────────────────────────────────
# 5. 统计汇总
# ──────────────────────────────────────────────

def summarize_warning(report: pd.DataFrame) -> dict:
    """汇总预警报告统计信息"""
    total   = len(report)
    summary = {}

    for level in ['HIGH', 'MEDIUM', 'LOW']:
        subset = report[report['risk_level'] == level]
        count  = len(subset)
        summary[level] = {
            'count'    : count,
            'ratio'    : count / total if total > 0 else 0,
            'avg_proba': subset['churn_proba'].mean() if count > 0 else 0,
        }

    summary['total']             = total
    summary['overall_avg_proba'] = report['churn_proba'].mean()

    print("\n" + "="*55)
    print("  客户流失预警汇总报告")
    print("="*55)
    print(f"  总客户数       : {total}")
    print(f"  平均流失概率   : {summary['overall_avg_proba']:.2%}")
    print(f"  {'-'*45}")
    print(f"  [高风险] 客户 : {summary['HIGH']['count']:5d} 人 "
          f"({summary['HIGH']['ratio']:.1%})  "
          f"均值={summary['HIGH']['avg_proba']:.2%}")
    print(f"  [中风险] 客户 : {summary['MEDIUM']['count']:5d} 人 "
          f"({summary['MEDIUM']['ratio']:.1%})  "
          f"均值={summary['MEDIUM']['avg_proba']:.2%}")
    print(f"  [低风险] 客户 : {summary['LOW']['count']:5d} 人 "
          f"({summary['LOW']['ratio']:.1%})  "
          f"均值={summary['LOW']['avg_proba']:.2%}")
    print("="*55)

    return summary


# ──────────────────────────────────────────────
# 6. 导出预警名单（增强版）
# ──────────────────────────────────────────────

def export_warning_list(report: pd.DataFrame,
                         output_dir: str = 'output',
                         risk_filter: str = None,
                         top_n: int = None,
                         fmt: str = 'csv') -> str:
    """
    导出预警名单到 CSV 或 Excel（按风险从高到低排序）

    参数:
        report     : build_warning_report 返回的 DataFrame（已排序）
        output_dir : 输出目录
        risk_filter: 仅导出指定风险等级 ('HIGH' / 'MEDIUM' / 'LOW' / None=全部)
        top_n      : 仅导出前 N 名高风险客户（None=全部）
        fmt        : 'csv' 或 'excel'

    返回:
        导出文件路径
    """
    os.makedirs(output_dir, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')

    df_export = report.copy()

    # 过滤风险等级
    if risk_filter and risk_filter.upper() in ['HIGH', 'MEDIUM', 'LOW']:
        df_export = df_export[df_export['risk_level'] == risk_filter.upper()]

    # 截取 Top N
    if top_n and top_n > 0:
        df_export = df_export.head(top_n)

    # 列名中文化（导出时更友好）
    col_rename = {
        'rank'          : '排名',
        'customerID'    : '客户ID',
        'customerName'  : '客户姓名',
        'gender'        : '性别',
        'phone'         : '联系电话',
        'email'         : '电子邮箱',
        'province'      : '省份',
        'city'          : '城市',
        'district'      : '区县',
        'address'       : '详细地址',
        'churn_proba'   : '流失概率',
        'risk_label_cn' : '风险等级',
        'tenure'        : '在网时长(月)',
        'MonthlyCharges': '月消费(元)',
        'TotalCharges'  : '累计消费(元)',
        'Contract'      : '合同类型',
        'InternetService': '网络服务',
        'PaymentMethod' : '支付方式',
        'strategy'      : '建议挽留措施',
        'top_risk_factors': '主要风险因素',
        'report_time'   : '报告生成时间',
        'churnReason'   : '流失原因分析',
    }
    # 只重命名存在的列
    existing_rename = {k: v for k, v in col_rename.items() if k in df_export.columns}
    df_export = df_export.rename(columns=existing_rename)

    # 丢弃不需要导出的内部列
    drop_cols = ['risk_level']
    df_export = df_export.drop(columns=[c for c in drop_cols if c in df_export.columns])

    # 流失概率格式化为百分比字符串
    if '流失概率' in df_export.columns:
        df_export['流失概率'] = df_export['流失概率'].apply(lambda x: f"{x:.1%}")

    suffix = f"_{risk_filter.lower()}" if risk_filter else "_all"
    suffix += f"_top{top_n}" if top_n else ""

    if fmt == 'excel':
        try:
            import openpyxl
            path = os.path.join(output_dir, f'warning_list{suffix}_{ts}.xlsx')
            with pd.ExcelWriter(path, engine='openpyxl') as writer:
                df_export.to_excel(writer, index=False, sheet_name='预警名单')
                # 自动调整列宽
                ws = writer.sheets['预警名单']
                for col in ws.columns:
                    max_len = max((len(str(cell.value)) for cell in col if cell.value), default=10)
                    ws.column_dimensions[col[0].column_letter].width = min(max_len + 4, 40)
            print(f"[导出] 预警名单(Excel)已保存至: {path}  ({len(df_export)} 条)")
        except ImportError:
            print("[提示] 未安装 openpyxl，自动降级为 CSV 格式")
            fmt = 'csv'

    if fmt == 'csv':
        path = os.path.join(output_dir, f'warning_list{suffix}_{ts}.csv')
        df_export.to_csv(path, index=False, encoding='utf-8-sig')
        print(f"[导出] 预警名单(CSV)已保存至: {path}  ({len(df_export)} 条)")

    return path


def export_report(report: pd.DataFrame, output_dir: str = 'output') -> str:
    """
    将完整预警报告导出为 CSV（兼容旧接口）

    参数:
        report    : 预警报告 DataFrame
        output_dir: 输出目录
    """
    return export_warning_list(report, output_dir=output_dir, fmt='csv')


# ──────────────────────────────────────────────
# 7. AI 接口：为高风险客户生成个性化挽留方案
# ──────────────────────────────────────────────

def _clean_ai_response(text: str) -> str:
    """
    清理AI返回内容中的对话格式标记和乱码，重新格式化结构
    """
    import re
    
    # 移除对话角色标记
    text = re.sub(r'\nuser\s*\n', '\n', text)
    text = re.sub(r'\nassistant\s*\n', '\n', text)
    text = re.sub(r'^user\s*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'^assistant\s*$', '', text, flags=re.MULTILINE)
    
    # 移除乱码字符（更严格的过滤）
    # 移除形如 !"#@$% 的单个乱码符号及其后的数字
    text = re.sub(r'[�]["\']?\*\*', '**', text)  # 修复 "** 乱码
    text = re.sub(r'[�][""''`\*]+', '', text)   # 移除含乱码的引号/星号
    text = re.sub(r'[�]["\']+', '', text)       # 移除单独的乱码引号
    
    # 移除常见的AI生成乱码模式
    text = re.sub(r'针对针对', '针对', text)
    text = re.sub(r'优先优先', '优先', text)
    text = re.sub(r'挽留挽留', '挽留', text)
    text = re.sub(r'方案方案', '方案', text)
    
    # 移除 Markdown 代码块标记
    text = re.sub(r'^```[\w]*\s*', '', text, flags=re.MULTILINE)
    text = re.sub(r'```\s*$', '', text, flags=re.MULTILINE)
    
    # 修复AI产生的错误编号（如 "1 1"、"2 2 2"、"1  2" 变成 "1.1"、"2.2"、"1.2"）
    # 将连续相同数字替换为单个数字+点
    text = re.sub(r'(\d)\s+(\d)', r'\1. \2', text)  # "1 1" -> "1. 1"
    text = re.sub(r'(\d)\s+(\d)\s+(\d)', r'\1. \2. \3', text)  # "1 1 1" -> "1. 1. 1"
    text = re.sub(r'(\d)\.(\d)\.(\d)\.(\d)', r'\1. \2. \3. \4', text)  # "1.1.1.1" -> "1. 1. 1. 1"
    
    # 移除百分比后面的乱码数字（如 "60.6%4" -> "60.6%"）
    text = re.sub(r'(\d+\.\d+)%\d+', r'\1%', text)
    text = re.sub(r'(\d+)%\d+', r'\1%', text)
    
    # 移除句子末尾的乱码数字
    text = re.sub(r'\s+\d+\s*$', '', text, flags=re.MULTILINE)
    
    # 修复 "优先优先度" 等重复词
    text = re.sub(r'([\u4e00-\u9fa5])\1{2,}', r'\1\1', text)  # 保留2个重复
    
    # 规范化编号格式
    lines = text.split('\n')
    cleaned_lines = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        # 移除行首的纯数字和点（除非是编号）
        line = re.sub(r'^(\d+)\s+(\d+\.)', r'\1. \2', line)
        # 确保标题有正确的格式
        if line.startswith('1.') or line.startswith('2.') or line.startswith('3.'):
            if '核心' in line or '流失原因' in line:
                line = '一、' + line[3:] if not line.startswith('一') else line
            elif '挽留' in line or '措施' in line:
                line = '二、' + line[3:] if not line.startswith('二') else line
            elif '优先级' in line or '建议' in line:
                line = '三、' + line[3:] if not line.startswith('三') else line
        cleaned_lines.append(line)
    
    # 规范化换行
    text = '\n'.join(cleaned_lines)
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    return text.strip()


def generate_ai_retention_plan(customer_id: str,
                                churn_proba: float,
                                risk_level: str,
                                churn_reasons: dict,
                                raw_features: dict = None,
                                api_key: str = None,
                                api_base: str = None,
                                model_name: str = 'gpt-3.5-turbo') -> str:
    """
    调用 AI 大模型接口，为高风险客户生成个性化挽留方案。

    兼容 OpenAI 格式（包括腾讯混元、讯飞星火、阿里通义等兼容接口）。
    若 api_key 未配置则返回规则兜底方案。

    参数:
        customer_id  : 客户 ID
        churn_proba  : 流失概率
        risk_level   : 风险等级
        churn_reasons: analyze_churn_reasons 的返回值
        raw_features : 客户原始特征字典（可选，提供给 AI 更多上下文）
        api_key      : API 密钥（None 则读取环境变量 OPENAI_API_KEY）
        api_base     : API 基础 URL（None 则使用官方接口）
        model_name   : 模型名称

    返回:
        str: AI 生成的个性化挽留方案文本
    """
    # ── 构造 Prompt ──
    top_reasons_text = '\n'.join([
        f"  {i+1}. {r['description']}（贡献度: {r['contribution']:.4f}）"
        for i, r in enumerate(churn_reasons.get('top_reasons', [])[:5])
    ])

    feature_text = ""
    if raw_features:
        key_fields = ['tenure', 'MonthlyCharges', 'Contract',
                      'InternetService', 'PaymentMethod',
                      'OnlineSecurity', 'TechSupport']
        lines = []
        for f in key_fields:
            if f in raw_features:
                cn = FEATURE_NAME_CN.get(f, f)
                lines.append(f"  - {cn}: {raw_features[f]}")
        feature_text = "\n客户基本信息:\n" + '\n'.join(lines)

    system_prompt = (
        "你是一名专业的电信客户运营专家，擅长客户挽留策略制定。"
        "请根据提供的客户流失分析数据，给出简洁、具体、可操作的个性化挽留方案。"
        "方案需包含：①核心流失原因分析（2-3条）②针对性挽留措施（3-5条）③优先级建议。"
        "语言简洁专业，避免泛泛而谈，确保措施与客户实际情况匹配。"
    )

    user_prompt = (
        f"客户ID: {customer_id}\n"
        f"流失概率: {churn_proba:.1%}  风险等级: {RISK_LABELS_CN.get(risk_level, risk_level)}\n"
        f"{feature_text}\n"
        f"\n流失风险因素分析（模型归因）:\n{top_reasons_text}\n"
        f"\n分析摘要: {churn_reasons.get('analysis_text', '')}\n"
        f"\n请生成针对该客户的个性化挽留方案："
    )

    # ── 尝试调用 AI 接口 ──
    _api_key = api_key if api_key else os.environ.get('OPENAI_API_KEY', '')
    _api_base = api_base if api_base else os.environ.get('OPENAI_API_BASE', 'https://api.openai.com/v1')

    if not _api_key:
        # 未配置 API Key，返回规则兜底方案
        return _fallback_retention_plan(customer_id, churn_proba, risk_level, churn_reasons)

    try:
        import urllib.request
        import urllib.error

        headers = {
            'Content-Type' : 'application/json',
            'Authorization': f'Bearer {_api_key}',
        }
        payload = json.dumps({
            'model'   : model_name,
            'messages': [
                {'role': 'system', 'content': system_prompt},
                {'role': 'user',   'content': user_prompt},
            ],
            'max_tokens': 800,
            'temperature': 0.7,
        }).encode('utf-8')

        url = _api_base.rstrip('/') + '/chat/completions'
        req = urllib.request.Request(url, data=payload, headers=headers, method='POST')

        with urllib.request.urlopen(req, timeout=30) as resp:
            result = json.loads(resp.read().decode('utf-8'))
            ai_text = result['choices'][0]['message']['content'].strip()
            
            # 清理AI返回内容中的对话格式标记和乱码
            ai_text = _clean_ai_response(ai_text)
            
            print(f"[AI挽留] 客户 {customer_id} 方案生成成功（{len(ai_text)} 字）")
            return ai_text

    except urllib.error.HTTPError as e:
        try:
            error_body = e.read().decode('utf-8')
            print(f"[AI挽留] HTTP {e.code} 错误: {error_body}")
        except:
            print(f"[AI挽留] HTTP {e.code} 错误")
        return _fallback_retention_plan(customer_id, churn_proba, risk_level, churn_reasons, raw_features)
    except Exception as e:
        print(f"[AI挽留] 接口调用失败: {e}，使用规则兜底方案")
        return _fallback_retention_plan(customer_id, churn_proba, risk_level, churn_reasons, raw_features)


def _fallback_retention_plan(customer_id: str,
                              churn_proba: float,
                              risk_level: str,
                              churn_reasons: dict,
                              raw_features: dict = None) -> str:
    """
    AI 接口未配置或调用失败时的规则兜底挽留方案
    根据流失原因和客户真实信息动态生成，更加个性化
    """
    strategies = RETENTION_STRATEGIES.get(risk_level, RETENTION_STRATEGIES['MEDIUM'])
    top_reasons = churn_reasons.get('top_reasons', [])
    analysis   = churn_reasons.get('analysis_text', '')

    # 提取客户真实信息用于个性化
    customer_name = raw_features.get('customerName', '客户') if raw_features else '客户'
    monthly = raw_features.get('MonthlyCharges', 0) if raw_features else 0
    tenure = raw_features.get('tenure', 0) if raw_features else 0
    contract = raw_features.get('Contract', '未知') if raw_features else '未知'
    internet = raw_features.get('InternetService', '未知') if raw_features else '未知'

    # 流失原因归因（从raw_features获取，更详细）
    churn_reason = ''
    if raw_features and 'churnReason' in raw_features and raw_features['churnReason']:
        churn_reason = raw_features['churnReason']
    elif analysis:
        churn_reason = analysis

    lines = [
        f"═══════════════════════════════════════════════",
        f"  客户姓名: {customer_name}",
        f"  客户ID: {customer_id}",
        f"═══════════════════════════════════════════════",
        f"",
        f"【风险概览】",
        f"  风险等级: {RISK_LABELS_CN.get(risk_level, risk_level)}",
        f"  流失概率: {churn_proba:.1%}",
        f"  在网时长: {tenure}个月",
        f"  月消费: ¥{monthly:.2f}",
        f"  合同类型: {contract}",
        f"  网络服务: {internet}",
        f"",
        f"【流失原因分析】",
    ]

    # 添加详细流失原因
    if churn_reason:
        if '；' in str(churn_reason):
            # 多个原因用分号分隔
            reasons_list = str(churn_reason).split('；')
            for i, r in enumerate(reasons_list, 1):
                if r.strip():
                    lines.append(f"  {i}. {r.strip()}")
        else:
            lines.append(f"  {churn_reason}")
    else:
        lines.append(f"  1. {analysis}" if analysis else "  暂无详细原因记录")

    # 基于原因生成针对性措施
    lines += ["", "【针对性挽留措施】"]

    # 根据具体情况添加个性化措施
    specific_measures = []

    if monthly > 80:
        specific_measures.append(f"★ 针对性优惠: 该客户月消费¥{monthly:.0f}偏高，建议提供10%-20%月租折扣优惠券（有效期30天）")

    if tenure < 12:
        specific_measures.append(f"★ 新客关怀: 入网{tenure}个月，属于新客户不稳定期，建议发送新人专属礼包领取通知")
    elif tenure < 24:
        specific_measures.append(f"★ 成长激励: 在网{tenure}个月，建议引导参与积分成长计划，增强黏性")

    if contract == 'Month-to-month':
        specific_measures.append(f"★ 合同升级: 建议主动联系推销年付/两年付合同，承诺更大折扣")

    if internet == 'Fiber optic':
        specific_measures.append(f"★ 网络质量保障: 光纤用户对质量敏感，建议主动询问网络体验，提供免费上门检测服务")

    if raw_features and raw_features.get('phone'):
        phone = raw_features['phone']
        specific_measures.append(f"★ 短信关怀: 发送关怀短信至{phone[:3]}****{phone[-4:]}，推送限时优惠活动")

    if specific_measures:
        for m in specific_measures:
            lines.append(f"  {m}")

    # 添加通用挽留措施
    lines += ["", "【通用挽留措施】"]
    lines += [f"  {s}" for s in strategies[:4]]  # 限制4条，避免太长

    lines += [
        "",
        f"【执行建议】",
        f"  1. 建议24小时内完成首次触达（电话/短信）",
        f"  2. 针对月消费>80元的客户，可申请专项折扣权限",
        f"  3. 记录客户反馈，完善客户画像档案",
        f"",
        f"═══════════════════════════════════════════════",
    ]

    return '\n'.join(lines)


# ──────────────────────────────────────────────
# 8. 批量 AI 挽留方案生成（针对高风险客户）
# ──────────────────────────────────────────────

def batch_ai_retention(report: pd.DataFrame,
                        X: np.ndarray,
                        model,
                        feature_names: list,
                        raw_df: pd.DataFrame = None,
                        top_n_customers: int = 10,
                        risk_level: str = 'HIGH',
                        api_key: str = None,
                        api_base: str = None,
                        model_name: str = 'Qwen/Qwen2.5-7B-Instruct',
                        output_dir: str = 'output') -> pd.DataFrame:
    """
    对指定风险等级的 Top N 客户批量生成 AI 挽留方案并导出报告

    参数:
        report          : build_warning_report 的返回值（已排序）
        X               : 全量特征矩阵
        model           : 已训练模型
        feature_names   : 特征名列表
        raw_df          : 原始数据 DataFrame（对应 X 的行顺序）
        top_n_customers : 处理前 N 名高风险客户
        risk_level      : 目标风险等级 ('HIGH'/'MEDIUM')
        api_key         : AI 接口 Key
        api_base        : AI 接口 Base URL
        output_dir      : 输出目录

    返回:
        DataFrame: 含 AI 挽留方案的增强报告
    """
    # 筛选目标客户
    target = report[report['risk_level'] == risk_level].head(top_n_customers).copy()

    risk_label = RISK_LABELS_CN.get(risk_level, risk_level)
    print(f"\n[AI] Starting AI retention plan for {len(target)} customer(s). Risk: {risk_label}")
    print("-" * 50)

    ai_plans = []
    reason_summaries = []

    # 建立 customerID → X 行索引的映射（通过 rank 列回溯原始位置）
    # report 的 rank 列从1开始，对应 sorted index，需要通过 customerID 匹配回 X
    all_ids = list(report['customerID'])

    for _, row in target.iterrows():
        cid  = row['customerID']
        prob = row['churn_proba']
        rlvl = row['risk_level']

        # 找到该客户在 X 中的原始行索引
        try:
            # report 是排序后的，rank 对应的原始索引需从全量 report 找回
            original_idx = all_ids.index(cid)
        except ValueError:
            original_idx = 0

        x_row = X[original_idx] if original_idx < len(X) else X[0]

        # 获取原始特征字典（直接从row获取，因为build_warning_report已经附加了这些字段）
        raw_feat = row.to_dict()

        # 确保raw_feat包含churnReason（从report的churnReason列）
        if 'churnReason' not in raw_feat or pd.isna(raw_feat.get('churnReason')):
            raw_feat['churnReason'] = row.get('churnReason', '')

        # 流失原因归因
        reasons = analyze_churn_reasons(
            x_row, model, feature_names,
            raw_customer_data=raw_feat, top_n=5
        )

        # 使用report中已有的详细流失原因（更完整）
        detailed_reason = row.get('churnReason', '') or reasons['analysis_text']
        reason_summaries.append(detailed_reason)

        # AI 生成挽留方案
        plan = generate_ai_retention_plan(
            cid, prob, rlvl, reasons, raw_feat,
            api_key=api_key, api_base=api_base, model_name=model_name
        )
        ai_plans.append(plan)

        # 打印带有客户姓名的日志
        name = raw_feat.get('customerName', cid)
        name_safe = str(name)[:20].encode('utf-8', errors='replace').decode('utf-8', errors='replace')
        reason_preview = str(detailed_reason)[:30].encode('utf-8', errors='replace').decode('utf-8', errors='replace')
        prob_pct = int(prob * 100)
        print(f"  [OK] {cid}({name_safe})  prob={prob_pct}%  reason: {reason_preview}...")

    target['churn_reason_summary'] = reason_summaries
    target['ai_retention_plan']    = ai_plans

    # 导出增强报告
    os.makedirs(output_dir, exist_ok=True)
    ts   = datetime.now().strftime('%Y%m%d_%H%M%S')
    path = os.path.join(output_dir, f'ai_retention_plans_{risk_level.lower()}_{ts}.csv')
    target.to_csv(path, index=False, encoding='utf-8-sig')
    print(f"\n[导出] AI 挽留方案报告已保存至: {path}")

    return target


# ──────────────────────────────────────────────
# 9. 单客户实时预警（API 调用场景）
# ──────────────────────────────────────────────

def single_customer_warning(customer_data: dict, model, scaler, encoders: dict,
                              feature_names: list,
                              api_key: str = None,
                              api_base: str = None,
                              model_name: str = 'Qwen/Qwen2.5-7B-Instruct') -> dict:
    """
    对单个客户实时计算流失风险，可选调用 AI 生成挽留方案

    参数:
        customer_data: 客户特征字典
        model        : 已训练模型
        scaler       : StandardScaler
        encoders     : LabelEncoder 字典
        feature_names: 特征名列表
        api_key      : AI 接口 Key（可选）
        api_base     : AI 接口 Base URL（可选）
        model_name   : 模型名称（可选）

    返回:
        dict: 包含流失概率、风险等级、原因分析、挽留建议
    """
    df = pd.DataFrame([customer_data])

    # 二值列编码
    binary_map = {'Yes': 1, 'No': 0, 'Male': 1, 'Female': 0}
    for col in ['gender', 'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']:
        if col in df.columns:
            df[col] = df[col].map(binary_map).fillna(df[col])

    # 多类别列编码
    for col, le in encoders.items():
        if col in df.columns:
            try:
                df[col] = le.transform(df[col].astype(str))
            except ValueError:
                df[col] = 0

    # 数值列标准化
    num_cols     = ['tenure', 'MonthlyCharges', 'TotalCharges']
    existing_num = [c for c in num_cols if c in df.columns]
    if existing_num:
        df[existing_num] = scaler.transform(df[existing_num])

    X          = df[feature_names].values
    proba      = predict_churn_proba(model, X)[0]
    risk_tuple = assign_risk_level(proba)
    risk_level = risk_tuple[0]  # 提取字符串代码
    risk_score = risk_tuple[1]  # 综合评分

    # 流失原因归因
    reasons = analyze_churn_reasons(
        X[0], model, feature_names,
        raw_customer_data=customer_data, top_n=5
    )

    # AI 挽留方案（或规则兜底）
    retention_plan = generate_ai_retention_plan(
        customer_data.get('customerID', 'DEMO'),
        proba, risk_level, reasons,
        raw_features=customer_data,
        api_key=api_key, api_base=api_base, model_name=model_name
    )

    result = {
        'churn_probability': round(float(proba), 4),
        'risk_level'       : risk_level,
        'risk_label'       : RISK_LABELS_CN[risk_level],
        'churn_reasons'    : reasons,
        'retention_plan'   : retention_plan,
        'strategies'       : RETENTION_STRATEGIES[risk_level],
        'report_time'      : datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    }

    print(f"\n{'─'*55}")
    print(f"[实时预警] 客户流失概率: {proba:.2%}  等级: {RISK_LABELS_CN[risk_level]}")
    print(f"\n[原因分析] {reasons['analysis_text']}")
    print(f"\n[主要风险因素]")
    for r in reasons['top_reasons'][:3]:
        print(f"  • {r['description']}  (贡献度={r['contribution']:.4f})")
    print(f"\n[挽留方案]")
    print(retention_plan)
    print(f"{'─'*55}")

    return result
