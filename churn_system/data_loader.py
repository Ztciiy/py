"""
=============================================================
模块: data_loader.py
功能: 数据加载与预处理模块
作者: 大四毕业设计项目
说明: 负责生成模拟数据集、加载真实/多份CSV数据集、数据清洗与预处理
=============================================================
"""

import os
import glob
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')


# ──────────────────────────────────────────────
# 0. 多 CSV 文件导入与合并
# ──────────────────────────────────────────────

# 系统期望的标准列名（可根据实际数据调整映射）
EXPECTED_COLUMNS = [
    'customerID', 'gender', 'SeniorCitizen', 'Partner', 'Dependents',
    'tenure', 'PhoneService', 'MultipleLines', 'InternetService',
    'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
    'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling',
    'PaymentMethod', 'MonthlyCharges', 'TotalCharges', 'Churn'
]


def load_single_csv(filepath: str, source_tag: str = None) -> pd.DataFrame:
    """
    加载单个 CSV 文件，并自动做基础列名校验

    参数:
        filepath  : CSV 文件路径
        source_tag: 来源标签（可选），添加到 source 列便于区分多数据集

    返回:
        DataFrame
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"[错误] 文件不存在: {filepath}")

    # 尝试多种编码读取
    for enc in ['utf-8', 'gbk', 'utf-8-sig', 'latin1']:
        try:
            df = pd.read_csv(filepath, encoding=enc)
            break
        except UnicodeDecodeError:
            continue
    else:
        raise ValueError(f"[错误] 无法识别文件编码: {filepath}")

    print(f"[加载] {os.path.basename(filepath)} → {df.shape[0]} 行 × {df.shape[1]} 列  (编码={enc})")

    # 自动列名模糊匹配（不区分大小写、下划线/空格互换）
    col_map = {}
    for col in df.columns:
        normalized = col.strip().replace(' ', '_').replace('-', '_')
        for expected in EXPECTED_COLUMNS:
            if normalized.lower() == expected.lower():
                col_map[col] = expected
                break
    if col_map:
        df.rename(columns=col_map, inplace=True)

    # 检查必要列
    required = {'Churn'}
    missing_required = required - set(df.columns)
    if missing_required:
        raise ValueError(f"[错误] 文件 {os.path.basename(filepath)} 缺少必要列: {missing_required}\n"
                         f"  当前列: {list(df.columns)}")

    # 添加来源标签
    tag = source_tag if source_tag else os.path.splitext(os.path.basename(filepath))[0]
    df['_source'] = tag

    return df


def load_multiple_csv(paths, dedup: bool = True) -> pd.DataFrame:
    """
    加载并合并多份 CSV 文件

    参数:
        paths : 文件路径列表，或包含 CSV 的文件夹路径（字符串）
                示例:
                  - ['data/a.csv', 'data/b.csv']
                  - 'data/'  （自动扫描文件夹下所有 .csv）
        dedup : 是否对 customerID 去重（默认 True）

    返回:
        合并后的 DataFrame
    """
    # 若传入文件夹路径，自动扫描
    if isinstance(paths, str):
        folder = paths
        csv_files = sorted(glob.glob(os.path.join(folder, '*.csv')))
        if not csv_files:
            raise FileNotFoundError(f"[错误] 文件夹 '{folder}' 下没有找到任何 .csv 文件")
        print(f"[扫描] 在 '{folder}' 发现 {len(csv_files)} 个 CSV 文件")
        paths = csv_files

    if not paths:
        raise ValueError("[错误] 未提供任何 CSV 文件路径")

    dfs = []
    for fp in paths:
        try:
            df = load_single_csv(fp)
            dfs.append(df)
        except Exception as e:
            print(f"[警告] 跳过文件 {fp}：{e}")

    if not dfs:
        raise RuntimeError("[错误] 所有文件加载失败，无法继续")

    # 合并
    merged = pd.concat(dfs, ignore_index=True, sort=False)
    print(f"\n[合并] 共加载 {len(dfs)} 份文件，合并后: {merged.shape[0]} 行 × {merged.shape[1]} 列")

    # 按 customerID 去重（保留第一条）
    if dedup and 'customerID' in merged.columns:
        before = len(merged)
        merged.drop_duplicates(subset='customerID', keep='first', inplace=True)
        removed = before - len(merged)
        if removed:
            print(f"[去重] 按 customerID 去除重复记录 {removed} 条，剩余 {len(merged)} 条")

    # 来源分布统计
    if '_source' in merged.columns:
        print("[来源统计]")
        for src, cnt in merged['_source'].value_counts().items():
            print(f"  {src}: {cnt} 条")

    # 流失率
    churn_col = 'Churn'
    if churn_col in merged.columns:
        churn_rate = (merged[churn_col].astype(str).str.lower().isin(['yes', '1', 'true'])).mean()
        print(f"[数据概览] 总样本: {len(merged)}，整体流失率: {churn_rate:.2%}")

    return merged


def load_data_auto(csv_paths=None, n_samples: int = 5000, random_state: int = 42) -> pd.DataFrame:
    """
    智能数据加载入口：有真实CSV则加载，否则生成模拟数据

    参数:
        csv_paths   : 文件路径列表 或 文件夹路径；为 None 时使用模拟数据
        n_samples   : 模拟数据量（仅在 csv_paths=None 时生效）
        random_state: 随机种子

    返回:
        DataFrame
    """
    if csv_paths is not None:
        print("[数据源] 使用真实 CSV 数据")
        return load_multiple_csv(csv_paths)
    else:
        print("[数据源] 未提供真实数据，使用模拟数据")
        return generate_mock_data(n_samples, random_state)


# ──────────────────────────────────────────────
# 1. 模拟数据生成（用于演示，无需真实数据集）
# ──────────────────────────────────────────────

# ── 真实姓名库 ──
SURNAMES = [
    '王', '李', '张', '刘', '陈', '杨', '赵', '黄', '周', '吴',
    '徐', '孙', '胡', '朱', '高', '林', '何', '郭', '马', '罗',
    '梁', '宋', '郑', '谢', '韩', '唐', '冯', '于', '董', '萧',
    '程', '曹', '袁', '邓', '许', '傅', '沈', '曾', '彭', '吕',
    '苏', '卢', '蒋', '蔡', '贾', '丁', '魏', '薛', '叶', '阎',
    '余', '潘', '杜', '戴', '夏', '钟', '汪', '田', '任', '姜',
    '范', '方', '石', '姚', '谭', '廖', '邹', '熊', '金', '陆',
    '郝', '孔', '白', '崔', '康', '毛', '邱', '秦', '江', '史',
]

GIVEN_NAMES = [
    '伟', '芳', '娜', '秀英', '敏', '静', '丽', '强', '磊', '军',
    '洋', '勇', '艳', '杰', '涛', '明', '超', '秀兰', '霞', '平',
    '刚', '桂英', '芬', '玲', '国华', '建华', '建国', '建军', '建军', '海',
    '志强', '永强', '鹏', '辉', '艳红', '艳芳', '秀珍', '秀英', '志明', '志刚',
    '文杰', '文华', '文彬', '晓东', '晓峰', '晓华', '晨光', '晨曦', '春华', '春梅',
    '秋香', '秋月', '冬梅', '夏雨', '夏莲', '冬兰', '玉兰', '玉珍', '玉英', '玉芬',
    '金兰', '银环', '翠兰', '翠花', '凤英', '凤兰', '凤珍', '桂花', '桂珍', '兰英',
    '丽华', '丽珍', '丽英', '丽芳', '婷', '雪梅', '雪莲', '雪花', '学文', '学武',
    '德明', '德强', '德志', '志远', '志刚', '志平', '志勇', '志红', '志军', '志国',
    '建华', '建平', '建新', '建东', '建伟', '海燕', '海涛', '海峰', '海东', '海亮',
]

# ── 手机号前缀（三大运营商号段）─
PHONE_PREFIXES = [
    '134', '135', '136', '137', '138', '139',  # 中国移动
    '147', '150', '151', '152', '157', '158', '159',  # 中国移动
    '172', '178', '182', '183', '184', '187', '188',  # 中国移动
    '198', '199',  # 中国移动新号段
    '130', '131', '132', '155', '156', '185', '186',  # 中国联通
    '145', '175', '176', '166',  # 中国联通
    '133', '153', '173', '177', '180', '181', '189',  # 中国电信
    '190', '191', '199',  # 中国电信
]

# ── 邮箱域名 ──
EMAIL_DOMAINS = [
    'qq.com', '163.com', '126.com', 'sina.com', 'sina.cn',
    'foxmail.com', 'hotmail.com', 'outlook.com', 'gmail.com',
    '139.com', '189.cn', 'wo.cn', 'aliyun.com',
]

# ── 城市信息 ──
CITIES = [
    ('北京', '北京市', '朝阳区'), ('北京', '北京市', '海淀区'), ('北京', '北京市', '东城区'),
    ('上海', '上海市', '浦东新区'), ('上海', '上海市', '徐汇区'), ('上海', '上海市', '黄浦区'),
    ('广州', '广东省广州市', '天河区'), ('广州', '广东省广州市', '白云区'), ('广州', '广东省广州市', '番禺区'),
    ('深圳', '广东省深圳市', '南山区'), ('深圳', '广东省深圳市', '龙岗区'), ('深圳', '广东省深圳市', '宝安区'),
    ('杭州', '浙江省杭州市', '西湖区'), ('杭州', '浙江省杭州市', '滨江区'), ('杭州', '浙江省杭州市', '余杭区'),
    ('成都', '四川省成都市', '武侯区'), ('成都', '四川省成都市', '锦江区'), ('成都', '四川省成都市', '青羊区'),
    ('武汉', '湖北省武汉市', '洪山区'), ('武汉', '湖北省武汉市', '江汉区'), ('武汉', '湖北省武汉市', '武昌区'),
    ('南京', '江苏省南京市', '鼓楼区'), ('南京', '江苏省南京市', '玄武区'), ('南京', '江苏省南京市', '建邺区'),
    ('西安', '陕西省西安市', '雁塔区'), ('西安', '陕西省西安市', '莲湖区'), ('西安', '陕西省西安市', '新城区'),
    ('重庆', '重庆市', '渝北区'), ('重庆', '重庆市', '江北区'), ('重庆', '重庆市', '南岸区'),
    ('天津', '天津市', '南开区'), ('天津', '天津市', '河西区'), ('天津', '天津市', '和平区'),
    ('苏州', '江苏省苏州市', '姑苏区'), ('苏州', '江苏省苏州市', '工业园区'), ('苏州', '江苏省苏州市', '虎丘区'),
    ('郑州', '河南省郑州市', '金水区'), ('郑州', '河南省郑州市', '二七区'), ('郑州', '河南省郑州市', '中原区'),
    ('长沙', '湖南省长沙市', '岳麓区'), ('长沙', '湖南省长沙市', '雨花区'), ('长沙', '湖南省长沙市', '天心区'),
    ('沈阳', '辽宁省沈阳市', '和平区'), ('沈阳', '辽宁省沈阳市', '皇姑区'), ('沈阳', '辽宁省沈阳市', '铁西区'),
    ('青岛', '山东省青岛市', '市南区'), ('青岛', '山东省青岛市', '崂山区'), ('青岛', '山东省青岛市', '黄岛区'),
    ('济南', '山东省济南市', '历下区'), ('济南', '山东省济南市', '市中区'), ('济南', '山东省济南市', '槐荫区'),
    ('福州', '福建省福州市', '鼓楼区'), ('福州', '福建省福州市', '台江区'), ('福州', '福建省福州市', '仓山区'),
    ('厦门', '福建省厦门市', '思明区'), ('厦门', '福建省厦门市', '湖里区'), ('厦门', '福建省厦门市', '集美区'),
]

# ── 详细地址 ──
STREETS = [
    '人民路', '中山路', '建设路', '解放路', '和平路', '长江路', '黄河路',
    '北京路', '上海路', '广州路', '深圳路', '杭州路', '南京路', '西安路',
    '科技路', '文化路', '商业街', '金融街', '开发区', '高新区', '工业园',
    '王府井大街', '南京路步行街', '春熙路', '观前街', '户部巷', '夫子庙',
]

# ── 流失原因映射（高风险客户）─
CHURN_REASONS = {
    'contract': '合同到期未续约',
    'price': '月租费用过高',
    'service': '网络质量不稳定',
    'competitor': '竞争对手更优惠',
    'unsatisfied': '服务不满意',
    'relocate': '搬迁/移居',
    'usage': '使用频率低',
}


def _generate_chinese_name(rng):
    """生成随机中文姓名"""
    surname = rng.choice(SURNAMES)
    given = rng.choice(GIVEN_NAMES)
    return surname + given


def _generate_phone(rng):
    """生成随机手机号"""
    prefix = rng.choice(PHONE_PREFIXES)
    suffix = ''.join([str(rng.integers(0, 10)) for _ in range(8)])
    return prefix + suffix


def _generate_email(name, rng):
    """基于姓名生成邮箱"""
    domain = rng.choice(EMAIL_DOMAINS)
    # 姓名转拼音/拼音首字母组合
    pinyin_names = {
        '伟': 'wei', '芳': 'fang', '娜': 'na', '秀': 'xiu', '英': 'ying',
        '敏': 'min', '静': 'jing', '丽': 'li', '强': 'qiang', '磊': 'lei',
        '军': 'jun', '洋': 'yang', '勇': 'yong', '艳': 'yan', '杰': 'jie',
        '涛': 'tao', '明': 'ming', '超': 'chao', '华': 'hua', '建': 'jian',
        '鹏': 'peng', '辉': 'hui', '婷': 'ting', '雪': 'xue', '梅': 'mei',
        '龙': 'long', '凤': 'feng', '林': 'lin', '海': 'hai', '志': 'zhi',
    }
    # 取前两个字的首字母/拼音
    p1 = ''.join([pinyin_names.get(c, c.lower()) for c in name[:2]])
    patterns = [
        f'{p1}{rng.integers(100, 1000)}',
        f'{p1}_{rng.integers(10, 100)}',
        f'{p1}{rng.choice(["88", "66", "99", "00"])}',
    ]
    return rng.choice(patterns) + '@' + domain


def generate_mock_data(n_samples: int = 5000, random_state: int = 42) -> pd.DataFrame:
    """
    生成模拟客户数据集（仿电信行业客户数据）

    参数:
        n_samples   : 样本数量，默认 5000 条
        random_state: 随机种子，保证可复现

    返回:
        DataFrame，包含客户特征与流失标签
    """
    np.random.seed(random_state)

    n = n_samples
    rng = np.random.default_rng(random_state)

    # ── 基础人口信息 ──
    gender         = rng.choice(['Male', 'Female'], n)
    senior_citizen = rng.choice([0, 1], n, p=[0.84, 0.16])
    partner        = rng.choice(['Yes', 'No'], n)
    dependents     = rng.choice(['Yes', 'No'], n)

    # ── 新增真实信息 ──
    names = [_generate_chinese_name(rng) for _ in range(n)]
    phones = [_generate_phone(rng) for _ in range(n)]
    emails = [_generate_email(name, rng) for name in names]
    cities = rng.choice(len(CITIES), n)
    city_info = [CITIES[i] for i in cities]
    provinces = [c[1] for c in city_info]
    districts = [c[2] for c in city_info]
    streets = [rng.choice(STREETS) for _ in range(n)]
    street_nums = [rng.integers(1, 1000) for _ in range(n)]
    addresses = [f'{province}市{district}{street}{num}号' for province, district, street, num in zip(provinces, districts, streets, street_nums)]

    # ── 服务使用信息 ──
    tenure              = rng.integers(1, 73, n)          # 在网月数
    phone_service       = rng.choice(['Yes', 'No'], n, p=[0.9, 0.1])
    multiple_lines      = rng.choice(['Yes', 'No', 'No phone service'], n, p=[0.4, 0.4, 0.2])
    internet_service    = rng.choice(['DSL', 'Fiber optic', 'No'], n, p=[0.34, 0.44, 0.22])
    online_security     = rng.choice(['Yes', 'No', 'No internet service'], n, p=[0.28, 0.5, 0.22])
    online_backup       = rng.choice(['Yes', 'No', 'No internet service'], n, p=[0.34, 0.44, 0.22])
    device_protection   = rng.choice(['Yes', 'No', 'No internet service'], n, p=[0.34, 0.44, 0.22])
    tech_support        = rng.choice(['Yes', 'No', 'No internet service'], n, p=[0.29, 0.49, 0.22])
    streaming_tv        = rng.choice(['Yes', 'No', 'No internet service'], n, p=[0.38, 0.40, 0.22])
    streaming_movies    = rng.choice(['Yes', 'No', 'No internet service'], n, p=[0.39, 0.39, 0.22])

    # ── 合同与账单信息 ──
    contract            = rng.choice(['Month-to-month', 'One year', 'Two year'], n, p=[0.55, 0.21, 0.24])
    paperless_billing   = rng.choice(['Yes', 'No'], n, p=[0.59, 0.41])
    payment_method      = rng.choice(
        ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'],
        n, p=[0.34, 0.23, 0.22, 0.21]
    )
    monthly_charges     = np.round(rng.uniform(18, 118, n), 2)
    total_charges       = np.round(monthly_charges * tenure + rng.normal(0, 50, n), 2)
    total_charges       = np.clip(total_charges, 0, None)  # 不允许负值

    # ── 构造流失标签（与特征相关，更真实） ──
    # 月租高、合同短、在网时间短 → 更易流失
    churn_prob = (
        0.05
        + 0.25 * (contract == 'Month-to-month').astype(float)
        + 0.10 * (internet_service == 'Fiber optic').astype(float)
        + 0.08 * (online_security == 'No').astype(float)
        + 0.08 * (tech_support == 'No').astype(float)
        - 0.15 * (tenure > 36).astype(float)
        + 0.05 * (monthly_charges > 75).astype(float)
    )
    churn_prob = np.clip(churn_prob, 0.02, 0.95)
    churn = np.where(rng.random(n) < churn_prob, 'Yes', 'No')

    # ── 为流失客户生成流失原因 ──
    churn_reason_text = [''] * n
    for i in range(n):
        if churn[i] == 'Yes':
            reasons = []
            if contract[i] == 'Month-to-month':
                reasons.append('合同到期未续约')
            if monthly_charges[i] > 80:
                reasons.append('月租费用过高')
            if internet_service[i] == 'Fiber optic':
                reasons.append('网络质量不稳定')
            if tech_support[i] == 'No':
                reasons.append('缺少技术支持服务')
            if online_security[i] == 'No' and internet_service[i] != 'No':
                reasons.append('缺少网络安全服务')
            if device_protection[i] == 'No' and internet_service[i] != 'No':
                reasons.append('缺少设备保护服务')
            if not reasons:
                reasons = ['其他原因']
            churn_reason_text[i] = '、'.join(reasons[:3])  # 最多3个原因

    # ── 组装 DataFrame ──
    df = pd.DataFrame({
        'customerID'        : [f'CUST-{str(i).zfill(5)}' for i in range(n)],
        'customerName'      : names,
        'gender'            : gender,
        'SeniorCitizen'     : senior_citizen,
        'Partner'           : partner,
        'Dependents'        : dependents,
        'phone'             : phones,
        'email'             : emails,
        'province'          : provinces,
        'city'              : [c[0] for c in city_info],
        'district'          : districts,
        'address'           : addresses,
        'tenure'            : tenure,
        'PhoneService'      : phone_service,
        'MultipleLines'     : multiple_lines,
        'InternetService'   : internet_service,
        'OnlineSecurity'    : online_security,
        'OnlineBackup'      : online_backup,
        'DeviceProtection'  : device_protection,
        'TechSupport'       : tech_support,
        'StreamingTV'       : streaming_tv,
        'StreamingMovies'   : streaming_movies,
        'Contract'          : contract,
        'PaperlessBilling'  : paperless_billing,
        'PaymentMethod'     : payment_method,
        'MonthlyCharges'    : monthly_charges,
        'TotalCharges'      : total_charges,
        'Churn'             : churn,
        'churnReason'       : churn_reason_text,  # 流失原因（仅流失客户有值）
    })

    print(f"[数据生成] 成功生成 {n} 条模拟客户数据")
    print(f"[数据生成] 流失率: {(df['Churn'] == 'Yes').mean():.2%}")
    print(f"[数据生成] 新增字段: customerName, phone, email, province, city, district, address, churnReason")
    return df


# ──────────────────────────────────────────────
# 2. 数据清洗与类型转换
# ──────────────────────────────────────────────

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    数据清洗：处理缺失值、异常值、类型转换

    参数:
        df: 原始 DataFrame

    返回:
        清洗后的 DataFrame
    """
    df = df.copy()

    # 处理 TotalCharges 列（可能含空字符串）
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

    # 用中位数填充数值型缺失值
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for col in num_cols:
        missing = df[col].isnull().sum()
        if missing > 0:
            median_val = df[col].median()
            df[col].fillna(median_val, inplace=True)
            print(f"[清洗] {col}: 填充 {missing} 个缺失值 (中位数={median_val:.2f})")

    # 用众数填充类别型缺失值
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()
    for col in cat_cols:
        missing = df[col].isnull().sum()
        if missing > 0:
            mode_val = df[col].mode()[0]
            df[col].fillna(mode_val, inplace=True)
            print(f"[清洗] {col}: 填充 {missing} 个缺失值 (众数={mode_val})")

    print(f"[清洗完成] 数据形状: {df.shape}，无缺失值: {df.isnull().sum().sum() == 0}")
    return df


# ──────────────────────────────────────────────
# 3. 特征编码与数值标准化
# ──────────────────────────────────────────────

def preprocess_features(df: pd.DataFrame):
    """
    特征编码与标准化：
      - 二值类别列 → 0/1 编码
      - 多类别列   → LabelEncoder 编码
      - 数值列     → StandardScaler 标准化

    参数:
        df: 清洗后的 DataFrame

    返回:
        X            : 特征矩阵 (numpy array)
        y            : 标签向量 (numpy array, 0/1)
        feature_names: 特征名列表
        scaler       : 已拟合的 StandardScaler（用于预测新样本时复用）
        encoders     : 各类别列的 LabelEncoder 字典
    """
    df = df.copy()

    # 删除无意义 ID 列和非模型特征列
    drop_cols = ['customerID', 'customerName', 'phone', 'email', 'province', 'city', 'district', 'address', 'churnReason', '_source']
    for col in drop_cols:
        if col in df.columns:
            df.drop(col, axis=1, inplace=True)

    # 目标变量
    y = (df['Churn'] == 'Yes').astype(int).values
    df.drop('Churn', axis=1, inplace=True)

    # ── 初始化编码器字典 ──
    encoders = {}

    # ── 二值列编码 (Male/Female, Yes/No) ──
    binary_map = {'Yes': 1, 'No': 0, 'Male': 1, 'Female': 0}
    binary_cols = ['gender', 'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']
    for col in binary_cols:
        if col in df.columns:
            df[col] = df[col].map(binary_map)
            # 处理可能的NaN（如果映射失败）
            if df[col].isnull().any():
                df[col] = df[col].fillna(0)

    # ── 多类别列编码 ──
    multi_cat_cols = [
        'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
        'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
        'Contract', 'PaymentMethod'
    ]
    for col in multi_cat_cols:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le

    # ── 确保所有列都是数值类型 ──
    for col in df.columns:
        if df[col].dtype == 'object' or df[col].dtype.name == 'category':
            # 未处理的类别列：编码
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
        else:
            # 数值列：填充NaN
            if df[col].isnull().any():
                df[col] = df[col].fillna(df[col].median())

    # ── 数值列标准化 ──
    num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    scaler = StandardScaler()
    df[num_cols] = scaler.fit_transform(df[num_cols])

    feature_names = df.columns.tolist()
    
    # 确保返回的X是float64类型且没有NaN
    X = df.values.astype(np.float64)
    
    # 再次检查并填充任何可能的NaN
    if np.isnan(X).any():
        col_medians = np.nanmedian(X, axis=0)
        for i in range(X.shape[1]):
            nan_mask = np.isnan(X[:, i])
            if nan_mask.any():
                X[nan_mask, i] = col_medians[i] if not np.isnan(col_medians[i]) else 0

    print(f"[预处理完成] 特征维度: {X.shape}，正样本(流失)比例: {y.mean():.2%}")
    return X, y, feature_names, scaler, encoders


# ──────────────────────────────────────────────
# 4. 数据集划分
# ──────────────────────────────────────────────

def split_dataset(X, y, test_size: float = 0.2, val_size: float = 0.1, random_state: int = 42):
    """
    将数据集划分为训练集、验证集、测试集

    比例默认: 训练 70% | 验证 10% | 测试 20%

    参数:
        X           : 特征矩阵
        y           : 标签向量
        test_size   : 测试集比例
        val_size    : 验证集比例（相对整体）
        random_state: 随机种子

    返回:
        X_train, X_val, X_test, y_train, y_val, y_test
    """
    # 先切出测试集
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    # 再从剩余中切出验证集
    val_ratio = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_ratio, random_state=random_state, stratify=y_temp
    )

    print(f"[数据划分] 训练集: {len(X_train)} | 验证集: {len(X_val)} | 测试集: {len(X_test)}")
    return X_train, X_val, X_test, y_train, y_val, y_test
