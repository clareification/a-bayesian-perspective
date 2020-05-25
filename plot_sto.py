import matplotlib.pyplot as plt 
import numpy as np 
import torch
import pickle as pkl 
import seaborn as sns

with open('data200.pkl', 'rb') as f:
    res_dict = pkl.load(f)
cs = sns.color_palette()
print( len(cs))

print(res_dict.keys())
fig, ax = plt.subplots()
#ax2 = ax.twinx()   
linestyles = ['solid', 'dashed', 'dotted', 'dashdot']
plt.title('Model selection for RFF lengthscales (n=200)')
ax.set_xlabel('approx log lengthscale')
ax.set_ylabel('likelihood (normalized by noise)')
for j, k in enumerate(list(res_dict.keys())[:4]):
    
    scale = -np.min(res_dict[k]['stos'][0])
    print(scale)
    for i, k2 in enumerate(list(res_dict[k].keys())):
        
        a = -1*np.log(k)/8/np.log(10) + 0.2
        
        #n = np.min(res_dict[k]['stos'])
        if k2 == 'stos' :
            p = [r[1] for r in res_dict[k][k2][0]]
            p = p if len(p) <= 9 else [p[:9]]+[p[9:]]
            p = np.array(p)
            sns.tsplot(p/scale, condition='method = ' + str(k2) + ' noise = ' + str(k),
             color=cs[j], alpha=1 , ax=ax, linestyle=linestyles[i])
        elif k2 not in ['ws', 'stos']:
            
            #axis = ax if k2 == 'mls' else ax2
            a =  res_dict[k][k2][0][:9]
            a = a if len(res_dict[k][k2][0])<=9 else a +[res_dict[k][k2][0][9:]]
            a = np.array(a)
            print(a)
            sns.tsplot(a/scale, condition=' method = ' + str(k2),
             color=cs[j], alpha = 1 , ax=ax, linestyle=linestyles[i])
ax.legend(bbox_to_anchor=(1.1, 1.0))
plt.tight_layout()
plt.savefig('stos2call' + '.png')
plt.clf()
