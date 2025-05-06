import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams
# 设置全局字体为Times New Roman
plt.rcParams['font.family'] = 'Times New Roman'

sample_ratios = [0.1, 0.2, 0.3, 0.4, 0.5]

M3DCNN_results = [87.78,91.51,91.78,93.48,94.92]
HybidSN_results = [91.93,95.64,97.08,97.73,97.48]
RNN_results = [83.31,87.9,91.24,93.76,92.62]
GCN_results = [91.56,94.38,95.19,96.17,96.75]
F2HGNNss_results = [92.83,95.2,95.67,97.15,97.01]
CEGCN_results = [94.52,96.59,97.22,97.64,98.4]
AMGCFN_results = [94.57,97.69,97.71,97.74,98.66]
AdvFNet_results = [95.49,97.03,98.03,98.69,98.8]


models = ['M3DCNN', 'HybidSN', 'RNN', 'GCN', 'F2HGNNss', 'CEGCN',"AMGCFN", 'DF2Net']

node_shapes = ['s', 'P', 'h', 'X', 'd', '*', 'v', '^']
plt.figure(figsize=(9, 7))

for i, model_results in enumerate([M3DCNN_results, HybidSN_results, RNN_results, GCN_results, F2HGNNss_results, CEGCN_results, AMGCFN_results,AdvFNet_results]):
    if models[i] == 'DF2Net':
        plt.plot(sample_ratios, model_results, marker=node_shapes[i], label=models[i], color='red', markersize=10, linewidth=3,linestyle='--')  # 设置线条和节点大小
    else:
        plt.plot(sample_ratios, model_results, marker=node_shapes[i], label=models[i], markersize=10, linewidth=3.0,linestyle='--')
plt.xlabel('Training Label Ratio(%)',fontsize='14')
plt.ylabel('OA(%)',fontsize='14')
plt.xticks(sample_ratios,fontsize='12')
plt.yticks(fontsize='12')
plt.legend(ncol=3, loc='lower right',fontsize='12', columnspacing=0.4)
plt.savefig("Diff_label_AdvFNet_XZ"+ '.png', format='png', transparent=True, dpi=400)
plt.show()
