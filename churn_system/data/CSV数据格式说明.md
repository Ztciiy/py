# CSV 数据格式说明

把你的客户数据做成 CSV 文件放到这个 `data/` 文件夹里，系统可以自动导入。

---

## 标准列名（推荐使用）

| 列名 | 类型 | 说明 | 示例值 |
|------|------|------|--------|
| customerID | 文本 | 客户唯一编号 | CUST-00001 |
| gender | 文本 | 性别 | Male / Female |
| SeniorCitizen | 数字 | 是否老年人 | 0 / 1 |
| Partner | 文本 | 是否有伴侣 | Yes / No |
| Dependents | 文本 | 是否有家属 | Yes / No |
| tenure | 数字 | 在网月数 | 1~72 |
| PhoneService | 文本 | 是否有电话服务 | Yes / No |
| MultipleLines | 文本 | 是否多线路 | Yes / No / No phone service |
| InternetService | 文本 | 网络类型 | DSL / Fiber optic / No |
| OnlineSecurity | 文本 | 是否有网络安全服务 | Yes / No / No internet service |
| OnlineBackup | 文本 | 是否有在线备份 | Yes / No / No internet service |
| DeviceProtection | 文本 | 是否有设备保护 | Yes / No / No internet service |
| TechSupport | 文本 | 是否有技术支持 | Yes / No / No internet service |
| StreamingTV | 文本 | 是否有流媒体TV | Yes / No / No internet service |
| StreamingMovies | 文本 | 是否有流媒体电影 | Yes / No / No internet service |
| Contract | 文本 | 合同类型 | Month-to-month / One year / Two year |
| PaperlessBilling | 文本 | 是否无纸化账单 | Yes / No |
| PaymentMethod | 文本 | 支付方式 | Electronic check / Mailed check / ... |
| MonthlyCharges | 数字 | 月消费金额 | 18.00~118.00 |
| TotalCharges | 数字 | 总消费金额 | 可为空 |
| **Churn** | 文本 | **流失标签（必填）** | **Yes / No** |

> ⚠️ **只有 `Churn` 列是必须的**，其他列缺失会自动用均值/众数填充。

---

## 如何导入你的数据

### 方法一：导入指定文件

打开 `main.py`，找到 `CONFIG` 字典，修改 `csv_paths`：

```python
'csv_paths': ['data/你的文件1.csv', 'data/你的文件2.csv'],
```

### 方法二：自动扫描文件夹

把所有 CSV 放进 `data/` 文件夹，然后设置：

```python
'csv_paths': 'data/',
```

系统会自动读取 `data/` 下所有 `.csv` 文件并合并。

### 方法三：使用模拟数据（默认）

```python
'csv_paths': None,
```

---

## CSV 文件示例

```
customerID,gender,SeniorCitizen,tenure,Contract,MonthlyCharges,TotalCharges,Churn
CUST-00001,Male,0,12,Month-to-month,75.5,906.0,Yes
CUST-00002,Female,0,36,One year,55.0,1980.0,No
CUST-00003,Male,1,5,Month-to-month,95.0,475.0,Yes
```

---

## 注意事项

1. 文件编码支持：UTF-8、GBK、UTF-8-BOM
2. 列名不区分大小写，空格和下划线可以互换
3. 多份数据合并时会按 `customerID` 自动去重
4. 数值列有空值会自动用中位数填充，类别列用众数填充
