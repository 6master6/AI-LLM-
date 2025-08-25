import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import seaborn as sns
import matplotlib.pyplot as plt

# 1. 读数据
df = pd.read_csv('./user_profiles_v2.csv')

# 2. 选取数值列并做缺失值处理 + 标准化
num_cols = df.select_dtypes(include=['int64','float32' ,'float64']).columns
X = df[num_cols].fillna(0)
print(X.head())

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. KMeans 聚类（K=3）
k = 3
kmeans = KMeans(n_clusters=k, random_state=42,n_init=20)
labels = kmeans.fit_predict(X_scaled)

# 把聚类结果写回原始 DataFrame
df['cluster'] = labels

# 4. 找到每个群体的“最具代表性用户”
representatives = []
for c in range(k):
    mask = labels == c
    idx_in_cluster = np.where(mask)[0]
    centroid = kmeans.cluster_centers_[c]
    # 计算所有成员到质心的欧氏距离
    distances = np.linalg.norm(X_scaled[mask] - centroid, axis=1)
    # 最近用户
    closest_idx = idx_in_cluster[np.argmin(distances)]
    rep_user = df.iloc[closest_idx].copy()
    representatives.append(rep_user)

rep_df = pd.DataFrame(representatives)
# rep_df.insert(0, 'cluster', range(k))
# rep_df = rep_df.reset_index(drop=True)

# 5. 打印代表用户
print("========== 3 位最具代表性的用户 ==========")
pd.set_option('display.max_columns', None)
print(rep_df)

# 6. 简单可视化（前两个标准化特征）
plt.figure(figsize=(6, 4))
sns.scatterplot(
    x=X_scaled[:, 0],
    y=X_scaled[:, 1],
    hue=labels,
    palette='Set2',
    alpha=0.6
)
sns.scatterplot(
    x=kmeans.cluster_centers_[:, 0],
    y=kmeans.cluster_centers_[:, 1],
    hue=range(k),
    palette='Set2',
    s=200,
    marker='X',
    ec='black',
    legend=False
)
plt.title('K-Means User Segmentation (top 2 features)')
plt.tight_layout()
plt.show()