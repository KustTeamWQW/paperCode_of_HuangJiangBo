import numpy as np
import scipy.io as scio
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import torch
# data= scio.loadmat(r'/home/ubuntu/code/CSGFNet/SAHGCN/SAHGCN/Datasets/paviaU.mat')["paviaU"]
# GT = scio.loadmat(r'/home/ubuntu/code/CSGFNet/SAHGCN/SAHGCN/Datasets/paviaU_gt.mat')["Data_gt"]
GT = scio.loadmat(r'/home/ubuntu/code/CSGFNet/SAHGCN/SAHGCN/Datasets/WHU_Hi_HongHu_gt.mat')["WHU_Hi_HongHu_gt"]
GT=GT[450:,:]
GT[GT > 5] -= 1

H, W= GT.shape
data=torch.load('out_all_WHU_SSRN.pt')
data=data.detach().cpu().numpy()
print(data.shape)
data=data.reshape(H, W, 19)
print(data.shape)
# 假设 data 是高光谱图像，GT 是标签矩阵
# data.shape = (H, W, C)
# GT.shape = (H, W)

num_samples_per_class = 500  # 每个类别选取的样本数

# 获取所有类别的标签
classes = np.unique(GT)
classes = classes[classes != 0]  # 排除背景类，如果有的话

# 初始化样本和标签列表
samples = []
labels = []

# 遍历每个类别，随机选择100个样本
for cls in classes:
    indices = np.argwhere(GT == cls)
    if len(indices) > num_samples_per_class:
        selected_indices = indices[np.random.choice(len(indices), num_samples_per_class, replace=False)]
    else:
        selected_indices = indices

    # 提取样本
    for idx in selected_indices:
        h, w = idx
        samples.append(data[h, w, :])
        labels.append(cls)

# 转换为 numpy 数组
samples = np.array(samples)
labels = np.array(labels)

# 使用 t-SNE 进行降维
tsne = TSNE(n_components=2, random_state=42)
tsne_results = tsne.fit_transform(samples)

# 设置字体为 Times New Roman
# plt.rcParams['font.family'] = 'Times New Roman'

# 可视化 t-SNE 结果并保存图像
plt.figure(figsize=(10, 8))
for cls in classes:
    idx = labels == cls
    plt.scatter(tsne_results[idx, 0], tsne_results[idx, 1], label=f'Class {cls}', alpha=0.7)

plt.legend()
# plt.title("t-SNE visualization of hyperspectral image features")
# plt.xlabel("t-SNE Component 1")
# plt.ylabel("t-SNE Component 2")

# 保存图像
output_filename = "tsne_visualization_all_WHU_SSRN.png"
plt.savefig(output_filename, format='png', dpi=400, bbox_inches='tight')
plt.show()

print(f"t-SNE visualization saved as {output_filename}")


