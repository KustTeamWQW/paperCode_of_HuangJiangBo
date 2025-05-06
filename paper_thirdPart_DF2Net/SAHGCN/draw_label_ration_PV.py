import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams
# 设置全局字体为Times New Roman
plt.rcParams['font.family'] = 'Times New Roman'

sample_ratios = [0.1, 0.2, 0.3, 0.4, 0.5]

M3DCNN_results = [80.36,83.4,87.51,88.56,90.93]
HybidSN_results = [76.84,87.18,87.59,92.21,94.01]
RNN_results = [72.23,76.23,72.86,78.01,80.83]
GCN_results = [86.44,88.92,90.66,89.46,91.25]
F2HGNNss_results = [89.2,91.04,90.89,91.29,94.26]
CEGCN_results = [90.47,94.23,95.27,96.67,97.03]
AMGCFN_results = [0.8987,0.94326293,0.9616206,0.9616053,0.9882159]
AMGCFN_results=np.array(AMGCFN_results) * 100
AdvFNet_results = [0.9419,0.96012855,0.968314,0.97632694,0.9893]
AdvFNet_results=np.array(AdvFNet_results) * 100

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
plt.savefig("Diff_label_AdvFNet_PV"+ '.png', format='png', transparent=True, dpi=400)
plt.show()
