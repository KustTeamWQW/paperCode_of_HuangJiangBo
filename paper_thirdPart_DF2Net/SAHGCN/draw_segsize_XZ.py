import matplotlib.pyplot as plt
from matplotlib import rcParams
import numpy as np
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = 'Times New Roman'

# 模擬一些實驗結果數據（這裡的數據是假設的，請根據實際數據進行替換）
segsize = [50,100,200,300,400]
S2HGCN_results = [0.9308,0.9254754,0.9073122,0.8902,0.8722]
S2HGCN_results=np.array(S2HGCN_results) * 100
AdvFNet_results = [0.9565734,0.9549826,0.95378953,0.9559186,0.956811]
AdvFNet_results=np.array(AdvFNet_results) * 100
# 畫出折綫圖
plt.plot(segsize, S2HGCN_results, marker='*',color='y', linestyle='--',linewidth=3, markersize=10,label='S2HGCN')
plt.plot(segsize, AdvFNet_results, marker='^', color='r',linestyle='--',linewidth=3, markersize=10,label='DF2Net')
# 添加標題和軸標籤
# plt.title('實驗結果折綫圖')
plt.legend(loc='lower left',fontsize='12')
plt.xlabel('Segmentation Scale',fontsize='14')
plt.ylabel('OA(%)',fontsize='14')
# 添加圖例

plt.savefig("seg_seize_XZ"+ '.png', format='png', transparent=True, dpi=400)
# 顯示折綫圖
plt.show()