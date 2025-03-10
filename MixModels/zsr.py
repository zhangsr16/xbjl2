import torch
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# 生成一个形状为 (26, 5, 4) 的示例张量
tensor_data = torch.randn(26, 5, 4)

# 将三维张量转换为二维数组
data_reshaped = tensor_data.reshape(-1, tensor_data.shape[-1])

# 使用K-Means聚类算法进行聚类分析
n_clusters = 3  # 设定聚类的数量
kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(data_reshaped)

# 获取聚类标签
labels = kmeans.labels_

# 将标签转换回原始张量的形状
labels_reshaped = labels.reshape(tensor_data.shape[:-1])

print("聚类标签：")
print(labels_reshaped)

# 使用PCA将数据降维到2D以便可视化
pca = PCA(n_components=2)
data_pca = pca.fit_transform(data_reshaped)

# 可视化聚类结果
plt.figure(figsize=(10, 7))
for i in range(n_clusters):
    cluster_data = data_pca[labels == i]
    plt.scatter(cluster_data[:, 0], cluster_data[:, 1], label=f'Cluster {i}')
plt.title('K-Means Clustering')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.legend()
plt.show()
print('end')