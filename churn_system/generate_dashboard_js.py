"""
生成仪表盘所需的 JS 数据文件（精简版）
"""
import pandas as pd
import os
import json
import re

def clean_ai_text(s):
    """清理AI生成的文本中的乱码"""
    if pd.isna(s) or s is None:
        return ''
    s = str(s)
    
    # 移除行首的单个大写字母后跟句号
    s = re.sub(r'\n[A-Z]\.', '\n', s)
    # 移除行首的 "Z" 或类似的乱码标记
    s = re.sub(r'\nZ+\s*', '\n', s)
    s = re.sub(r'\n\d+\.\s*', '\n', s)
    s = re.sub(r'\n11+\.\s*', '\n', s)
    s = re.sub(r'\n1+\.\s*', '\n', s)
    s = re.sub(r'\n2+\.\s*', '\n', s)
    
    # 移除 "贡献度" 后面的乱码数字和括号内容
    s = re.sub(r'（[^）]*%[^）]*）', '', s)
    s = re.sub(r'贡献度[：:]\s*[\d.%,]+', '', s)
    
    # 移除末尾的逗号、句号和特殊字符
    s = re.sub(r'，+\s*$', '', s)
    s = re.sub(r'\.\s*$', '', s)
    s = re.sub(r'，+', '，', s)
    
    # 移除所有连续的 "Z" 字符（这是乱码）
    s = re.sub(r'Z{2,}', '', s)
    
    # 移除特殊乱码字符
    s = re.sub(r'\s+', ' ', s)
    s = re.sub(r'[\+\-]+', '', s)
    s = re.sub(r'^\s*n\s*', '', s, flags=re.MULTILINE)  # 移除开头的 "n"
    
    # 修复重复字符问题
    s = re.sub(r'针对针对', '针对', s)
    s = re.sub(r'优先优先', '优先', s)
    s = re.sub(r'挽留挽留', '挽留', s)
    s = re.sub(r'方案方案', '方案', s)
    s = re.sub(r'服务服务', '服务', s)
    s = re.sub(r'客户客户', '客户', s)
    s = re.sub(r'合同合同', '合同', s)
    s = re.sub(r'二二', '二', s)
    s = re.sub(r'一一', '一', s)
    s = re.sub(r'三三', '三', s)
    s = re.sub(r'四四', '四', s)
    s = re.sub(r'对对', '对', s)
    s = re.sub(r'可能可能', '可能', s)
    s = re.sub(r'问题问题', '问题', s)
    
    # 修复乱码的 "三、" 开头
    s = re.sub(r'三、三、', '三、', s)
    s = re.sub(r'三、三、', '三、', s)
    s = re.sub(r'二、二、', '二、', s)
    s = re.sub(r'\n三、三、', '\n三、', s)
    s = re.sub(r'\n二、二、', '\n二、', s)
    
    # 移除只有单个字符的行
    lines = s.split('\n')
    lines = [line.strip() for line in lines if len(line.strip()) > 3]
    s = '\n'.join(lines)
    
    # 最后一次清理
    s = re.sub(r' +', ' ', s)
    s = re.sub(r'\n+', '\n', s)
    
    return s.strip()

def escape_js_string(s):
    """将字符串转义为JavaScript安全格式"""
    if pd.isna(s) or s is None:
        return ''
    s = clean_ai_text(s)
    s = str(s)
    s = s.replace('\\', '\\\\')
    s = s.replace('\n', '\\n')
    s = s.replace('\r', '\\r')
    s = s.replace('"', '\\"')
    s = s.replace("'", "\\'")
    s = s.replace('\t', '\\t')
    return s

def generate_dashboard_js():
    output_dir = 'output'

    # 获取最新的文件
    warning_files = [f for f in os.listdir(output_dir) if f.startswith('warning_list_all_') and f.endswith('.csv')]
    ai_files = [f for f in os.listdir(output_dir) if f.startswith('ai_retention_plans_') and f.endswith('.csv')]

    if not warning_files:
        print("[错误] 未找到预警名单文件")
        return

    # 使用最新的文件
    warning_file = sorted(warning_files)[-1]
    ai_file = sorted(ai_files)[-1] if ai_files else None

    print(f"[数据] 加载预警名单: {warning_file}")
    warning_df = pd.read_csv(os.path.join(output_dir, warning_file), encoding='utf-8-sig')

    # 生成精简版 WARNING_DATA.js
    js_data = []
    for _, row in warning_df.iterrows():
        prob = row.get('流失概率', row.get('churn_proba', 0))
        if isinstance(prob, str):
            prob = float(prob.replace('%', '')) / 100
        else:
            prob = float(prob)

        level = str(row.get('风险等级', row.get('risk_level', 'LOW')))
        if '高' in level or level == 'HIGH':
            level = 'HIGH'
        elif '中' in level or level == 'MEDIUM':
            level = 'MEDIUM'
        else:
            level = 'LOW'

        customer_id = str(row.get('客户ID', row.get('customerID', 'UNKNOWN')))

        churn_reason = str(row.get('流失原因分析', row.get('churnReason', '')))
        if churn_reason == 'nan':
            churn_reason = ''
        if len(churn_reason) > 100:
            churn_reason = churn_reason[:100] + '...'

        item = {
            'id': customer_id,
            'n': customer_id[:6],
            'prob': round(prob, 4),
            'lv': level,
            'rs': escape_js_string(churn_reason),
            'ct': row.get('合同类型', row.get('Contract', '-')),
            'pm': row.get('支付方式', row.get('PaymentMethod', '-')),
            'm': round(float(row.get('月消费(元)', row.get('MonthlyCharges', 0))) if pd.notna(row.get('月消费(元)', row.get('MonthlyCharges', 0))) else 0, 2),
            't': round(float(row.get('累计消费(元)', row.get('TotalCharges', 0))) if pd.notna(row.get('累计消费(元)', row.get('TotalCharges', 0))) else 0, 2),
        }
        js_data.append(item)

    warning_js_path = os.path.join(output_dir, 'WARNING_DATA.js')
    with open(warning_js_path, 'w', encoding='utf-8') as f:
        f.write('const WARNING_DATA=')
        f.write(json.dumps(js_data, ensure_ascii=False))
        f.write(';')
    print(f"[生成] {warning_js_path} ({len(js_data)} 条记录)")

    # 生成 AI_DATA.js
    if ai_file:
        print(f"[数据] 加载AI挽留方案: {ai_file}")
        ai_df = pd.read_csv(os.path.join(output_dir, ai_file), encoding='utf-8-sig')

        ai_data = []
        for _, row in ai_df.iterrows():
            prob = float(row.get('churn_proba', 0))
            customer_id = str(row.get('customerID', 'UNKNOWN'))

            reason = str(row.get('churn_reason_summary', row.get('churnReason', '')))
            if reason == 'nan':
                reason = str(row.get('churnReason', ''))
            if reason == 'nan':
                reason = ''
            reason = clean_ai_text(reason)

            plan = str(row.get('ai_retention_plan', ''))
            if plan == 'nan':
                plan = ''
            plan = clean_ai_text(plan)

            item = {
                'rk': int(row.get('rank', 0)),
                'id': customer_id,
                'n': customer_id[:6],
                'prob': round(prob, 4),
                'rs': escape_js_string(reason),
                'pl': escape_js_string(plan),
                'ct': row.get('Contract', '-'),
            }
            ai_data.append(item)

        ai_js_path = os.path.join(output_dir, 'AI_DATA.js')
        with open(ai_js_path, 'w', encoding='utf-8') as f:
            f.write('const AI_DATA=')
            f.write(json.dumps(ai_data, ensure_ascii=False))
            f.write(';')
        print(f"[生成] {ai_js_path} ({len(ai_data)} 条记录)")
    else:
        ai_js_path = os.path.join(output_dir, 'AI_DATA.js')
        with open(ai_js_path, 'w', encoding='utf-8') as f:
            f.write('const AI_DATA=[];')
        print(f"[生成] {ai_js_path} (空数据)")

    print("[完成] 仪表盘数据文件生成完毕")

if __name__ == '__main__':
    generate_dashboard_js()