# user_embedding_pca.py
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA


# --------------------------------------------------
# 0. 设置代理
# --------------------------------------------------
os.environ['http_proxy'] = 'socks5h://127.0.0.1:1080'
os.environ['https_proxy'] = 'socks5h://127.0.0.1:1080'

# --------------------------------------------------
# 1. 读数据
# --------------------------------------------------
df = pd.read_csv('user_profiles_v2.csv')
df['consumption_3'] = df['consumption'].replace({
    '中低': '中',
    '中高': '中'
})

# --------------------------------------------------
# 2. 把类别特征拼成一句话
# --------------------------------------------------
cols_to_use = ['sex', 'city', 'consumption', 'os', 'payment', 'interests']
df['text'] = df[cols_to_use].astype(str).agg('，'.join, axis=1)

# --------------------------------------------------
# 3. 用 BGE-M3 编码（768 维）
# --------------------------------------------------
model = SentenceTransformer('BAAI/bge-base-zh-v1.5', device='cuda')   # 显存小就 cpu
embeddings = model.encode(df['text'], show_progress_bar=True, normalize_embeddings=True)
# embeddings.shape == (500, 768)

# --------------------------------------------------
# 4. PCA 降维 → 2 维
# --------------------------------------------------
pca = PCA(n_components=2, random_state=42)
coords = pca.fit_transform(embeddings)
df['pca1'] = coords[:, 0]
df['pca2'] = coords[:, 1]

# --------------------------------------------------
# 5. 可视化
# --------------------------------------------------
plt.rcParams["font.sans-serif"] = ["SimHei"] 
plt.rcParams["axes.unicode_minus"] = False
plt.figure(figsize=(8, 6))
palette = {'高': 'red', '中': 'orange', '低': 'green'}
sns.scatterplot(data=df, x='pca1', y='pca2', hue='consumption_3',
                palette=palette, s=60, alpha=0.8)
plt.title('500 位用户的 BGE-M3 Embedding → PCA 2D 可视化')
plt.xlabel('PCA-1')
plt.ylabel('PCA-2')
plt.legend(title='消费水平')
plt.tight_layout()
plt.show()