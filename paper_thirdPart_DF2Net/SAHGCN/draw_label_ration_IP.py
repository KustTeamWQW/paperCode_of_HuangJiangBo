import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams
# 设置全局字体为Times New Roman
plt.rcParams['font.family'] = 'Times New Roman'

sample_ratios = [1, 2, 3, 4, 5]

M3DCNN_results = [61.36,66.51,80.3,83.47,85.83]
HybidSN_results = [78.8,90.36,88.41,93.19,92.36]
RNN_results = [53.96,61.51,62.95,69.64,67.87]
GCN_results = [79.84,88.53,90.29,92.04,92.27]
F2HGNNss_results = [85.83,89.98,91.63,92.49,92.22]
CEGCN_results = [77.58,89.08,94.68,96.86,97.49]
AMGCFN_results = [85.18,94.28,94.02,95.66,97.39]

AdvFNet_results = [87.06,94.61,96.43,97.46,97.8]


models = ['M3DCNN', 'HybidSN', 'RNN', 'GCN', 'F2HGNNss', 'CEGCN',"AMGCFN", 'DF2Net']

node_shapes = ['s', 'P', 'h', 'X', 'd', '*', 'v', '^']
plt.figure(figsize=(9, 7))

for i, model_results in enumerate([M3DCNN_results, HybidSN_results, RNN_results, GCN_results, F2HGNNss_results, CEGCN_results, AMGCFN_results,AdvFNet_results]):
    if models[i] == 'DF2Net':
        plt.plot(sample_ratios, model_results, marker=node_shapes[i], label=models[i], color='red', markersize=10, linewidth=3,linestyle='--')  # 设置线条和节点大小
    else:
        plt.plot(sample_ratios, model_results, marker=node_shapes[i], label=models[i], markersize=10, linewidth=3.0,linestyle='--')
# plt.title(' Indian Pins',fontsize='11')
plt.xlabel('Training Label Ratio(%)',fontsize='14')
plt.ylabel('OA(%)',fontsize='14')
plt.xticks(sample_ratios,fontsize='12')
plt.yticks(fontsize='12')
plt.legend(ncol=3, loc='lower right',fontsize='12', columnspacing=0.4)
plt.savefig("Diff_label_AdvFNet_IP"+ '.png', format='png', transparent=True, dpi=400)
plt.show()
