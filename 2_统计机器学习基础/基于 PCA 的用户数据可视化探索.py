#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PCA 降维 + K-Means 聚类联动可视化
依赖：pandas, scikit-learn, matplotlib, seaborn
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

# 1. 读取数据
df = pd.read_csv('user_profiles_v2.csv')

# 2. 数值特征提取、缺失填补、标准化
num_cols = df.select_dtypes(include=['int64', 'float64']).columns
X = df[num_cols].fillna(0)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. K-Means 聚类（K=3），生成标签
k = 3
kmeans = KMeans(n_clusters=k, random_state=42)
df['cluster'] = kmeans.fit_predict(X_scaled)

# 4. PCA 降维
pca_2d = PCA(n_components=2, random_state=42)
pcs_2d = pca_2d.fit_transform(X_scaled)

pca_3d = PCA(n_components=3, random_state=42)
pcs_3d = pca_3d.fit_transform(X_scaled)

# 5. 调色板
palette = sns.color_palette('Set2', k)
cluster_colors = [palette[c] for c in df['cluster']]

# 6. 2D 散点图
plt.figure(figsize=(7, 5))
sns.scatterplot(x=pcs_2d[:, 0], y=pcs_2d[:, 1],
                hue=df['cluster'], palette=palette, s=40, alpha=0.8)
plt.title('PCA 2D — colored by K-Means cluster (k=3)')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend(title='Cluster')
plt.tight_layout()
plt.show()

# 7. 3D 散点图
from mpl_toolkits.mplot3d import Axes3D   # noqa: F401  仅用于激活 3D
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
for c in range(k):
    mask = df['cluster'] == c
    ax.scatter(pcs_3d[mask, 0],
               pcs_3d[mask, 1],
               pcs_3d[mask, 2],
               color=palette[c],
               label=f'Cluster {c}',
               s=40,
               alpha=0.8)
ax.set_title('PCA 3D — colored by K-Means cluster (k=3)')
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_zlabel('PC3')
ax.legend()
plt.tight_layout()
plt.show()

# 8. 可选：查看各主成分解释方差
print('Explained variance ratio (2D):', pca_2d.explained_variance_ratio_)
print('Explained variance ratio (3D):', pca_3d.explained_variance_ratio_)