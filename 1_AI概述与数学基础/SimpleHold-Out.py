import pandas as pd
from sklearn.model_selection import train_test_split

# 1. 读入数据
df = pd.read_csv('user_profiles_v2.csv')

# 2. 简化为 高/中/低 三类
df['consumption_3'] = df['consumption'].replace({
    '中低': '中',
    '中高': '中'
})

# 3. “黄金标准”——原始分布 
# .value_counts() - 频次统计
#功能：统计每个唯一值的出现次数
#参数解析：
#normalize=True：返回比例而非绝对数量
# 默认normalize=False会返回计数

#.sort_index() - 索引排序
#功能：按索引（即类别标签）字母顺序排序
#重要性：确保结果顺序一致，便于比较
gold = df['consumption_3'].value_counts(normalize=True).sort_index()
print("黄金标准（全量 500 条）:\n", gold.round(3), '\n')

# 4. 随机 80/20 划分（stratify=None，即“有风险”的简单随机）关键设置 - 不进行分层抽样
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=None)

# 5. 分别在训练集、测试集上统计
train_dist = train_df['consumption_3'].value_counts(normalize=True).sort_index()
test_dist  = test_df['consumption_3'].value_counts(normalize=True).sort_index()

print("训练集分布（400 条）:\n", train_dist.round(3), '\n')
print("测试集分布（100 条）:\n", test_dist.round(3), '\n')

# 6. 计算绝对差值（与黄金标准比较）
train_diff = (train_dist - gold).abs()
test_diff  = (test_dist  - gold).abs()

print("训练集 | 与黄金标准绝对差:\n", train_diff.round(3), '\n')
print("测试集 | 与黄金标准绝对差:\n", test_diff.round(3), '\n')