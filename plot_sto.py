import matplotlib.pyplot as plt 
import numpy as np 
import torch
import pickle as pkl 
import seaborn as sns

# Plot sample-then-optimize data
with open('data200.pkl', 'rb') as f:
    res_dict = pkl.load(f)
cs = sns.color_palette()
print( len(cs))
font = {'family':'serif', 'size':16}
plt.rc('font', **font)
legend_dict={'stos':'Sample-then-optimize', 'els':'Exact ELBO', 'mls':'Log Evidence'}
print(res_dict.keys())
fig, axs = plt.subplots(1, 3, figsize=(14, 4))
#ax2 = ax.twinx()   
linestyles = ['solid', 'dashed', 'dotted', 'dashdot']
for ax in axs:
    ax.set_xlabel('Log Lengthscale')
axs[0].set_ylabel('Log Likelihood (normalized)')
for j, k in enumerate(list(res_dict.keys())[::2]):
    
    scale = -np.min(res_dict[k]['stos'][0])
    #print(scale)
    print(res_dict[k].keys())
    handles = []
    labels =[]
    for i, k2 in enumerate(list(res_dict[k].keys())):
        
        a = -1*np.log(k)/8/np.log(10) + 0.2
        
        #n = np.min(res_dict[k]['stos'])
        if k2 == 'stos' :
            print(k2)
            p = [r[1] for r in res_dict[k][k2][0]]
            p = p if len(p) <= 9 else [p[:9]]+[p[9:]]
            p = np.array(p)
            sns.tsplot(p/scale, condition=legend_dict[k2] + r', $\sigma^2=$' + str(k),
             color=cs[j], alpha=1 , ax=axs[j], linestyle=linestyles[i])
        elif k2 not in ['ws', 'stos']:
            
            #axis = ax if k2 == 'mls' else ax2
            a =  res_dict[k][k2][0][:9]
            print(k2, '\n')
            #a = a if len(res_dict[k][k2][0])<=9 else a +[res_dict[k][k2][0][9:]]
            a = np.array(a)
            #print(a)
            sns.tsplot(np.array(np.real(a))/scale, condition=legend_dict[k2] + r', $\sigma^2=$' + str(k),
             color=cs[j], alpha = 1 , ax=axs[j], linestyle=linestyles[i])
    h, l = axs[j].get_legend_handles_labels()
    handles.append(h)
    labels.append(l)
    axs[j].legend(fontsize=10)
    # if j < 2:
    #     axs[j].get_legend().remove()
    
#fig.legend(handles , labels , bbox_to_anchor=(1.1, 1.0))
fig.suptitle('RFF Frequency Selection')
#ax.legend()
plt.tight_layout()
plt.savefig('stos2callmay29' + '.png')
plt.clf()
